#!/usr/bin/env python
# Copyright 2024-2025 The Alibaba 3DAIGC Team Authors. All rights reserved.
"""
prepare_4ddress.py — 将 4DDress 数据集转换为 LHM 推理/训练所需格式。

=== 核心设计决策 ===

[SMPL-X 参数选择]
  LHM 训练数据（ClothVideo）的 SMPL-X 由 Multi-HMR 估计，输出在"虚拟相机空间"
  （FOV=60°，focal 由图像尺寸和视角推算）。

  4DDress 提供多视角优化的高质量 SMPL-X（世界坐标系，真实标定相机参数）。

  本脚本选择：使用 4DDress 自带的 SMPL-X，理由：
    1. 精度远高于 Multi-HMR 单目估计（多视角 GT 优化）
    2. 4DDress 本身即多视角数据集，坐标系来源清晰
    3. 避免对已有 GT 重复跑重型模型

  关键转换：将世界坐标系 SMPL-X (global_orient, transl)
            转换到前视角相机坐标系，并使用真实标定内参 (K) 填写 focal/princpt。

[FLAME 参数]
  4DDress 不含 FLAME 面部参数，必须通过 LHM_Track 的 gaga_track 流程估计。
  这需要先运行 Sapiens 2D 关键点，再运行 gaga_track FLAME 拟合。

[输出结构]
  {output_dir}/{subject}_{outfit}_{take}/
  ├── smplx_params/           # SMPL-X in 前视角相机坐标系（格式与 video2motion 一致）
  │   └── {frame:05d}.json
  ├── cameras.json            # 所有 4 路相机的内参/外参/c2w（供训练 Dataset）
  │
  │   # 以下三项仅 FLAME 估计流程需要，训练 Dataset 不读取：
  ├── imgs_png/               # 指向原始图像的软链接（gaga_track 输入，1-based 命名，不占额外磁盘）
  │   └── {frame:05d}.png    # → /data/4ddress/{subject}/{outfit}/{take}/Capture/{cam}/images/...
  ├── sapiens_pose/           # Sapiens 2D 关键点（gaga_track 依赖）
  │   └── {frame:05d}.json
  └── flame_params/           # FLAME bbox（head crop 用）
      └── {frame:05d}.json

  # --build_meta 生成，供 LHM BaseDataset._load_uids() 读取：
  {output_dir}/label/
  ├── train_list.json         # 训练集 UID 列表，每条为 "subject_outfit_take"
  └── val_list.json

用法示例：
  # 处理单个序列（只转 SMPL-X，不跑 FLAME）
  python prepare_4ddress.py \\
      --root_dir /path/to/4ddress \\
      --subject 00122 --outfit Outer --take Take1 \\
      --output_dir ./train_data/4ddress_lhm \\
      --mode smplx

  # 在已转换的序列上补充跑 FLAME 估计
  python prepare_4ddress.py \\
      --root_dir /path/to/4ddress \\
      --output_dir ./train_data/4ddress_lhm \\
      --mode flame --model_path ./pretrained_models

  # 完整流程（SMPL-X 转换 + FLAME 估计）
  python prepare_4ddress.py \\
      --root_dir /path/to/4ddress \\
      --output_dir ./train_data/4ddress_lhm \\
      --mode all --sample_rate 5 --build_meta
"""

import argparse
import glob
import json
import os
import pickle
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# SMPLXDecoder 可选依赖（需要 torch + smplx，即 LHM 已依赖的官方 smplx 包）
_SMPLX_DECODER_AVAILABLE = False
try:
    import torch
    import smplx
    _SMPLX_DECODER_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

CAMERA_VIEWS = ['0004', '0028', '0052', '0076']  # 4DDress 固定 4 路相机 ID
LHM_BETA_DIM = 10     # LHM 使用 10 维 shape
LHM_EXPR_DIM = 100    # LHM 使用 100 维 expression（对应 FLAME）


# ===========================================================================
# 1. 相机工具函数
# ===========================================================================

def load_cameras_pkl(cameras_pkl_path: str) -> dict:
    """加载 4DDress cameras.pkl，返回 {cam_id: {'intrinsics': ..., 'extrinsics': ...}}"""
    with open(cameras_pkl_path, 'rb') as f:
        return pickle.load(f)


def get_sorted_camera_views(item_dir: str, cameras: dict,
                             camera_views: list = CAMERA_VIEWS) -> list:
    """
    按照人体正面方向排序相机视角，返回有序列表（前视角第一）。
    复用 dress4d.py 中 sort_camera_views 的逻辑。
    """
    # 读取第一帧的 SMPL global_orient 确定人体正面方向
    smpl_files = sorted(glob.glob(os.path.join(item_dir, 'SMPL', '*.pkl')))
    if not smpl_files:
        smpl_files = sorted(glob.glob(os.path.join(item_dir, 'SMPLX', '*.pkl')))
    if not smpl_files:
        return [c for c in camera_views if c in cameras]

    smpl_data = pickle.load(open(smpl_files[0], 'rb'))
    root_orient = np.array(smpl_data['global_orient'], dtype=np.float32).reshape(3)
    rotation_matrix = Rotation.from_rotvec(root_orient).as_matrix()
    front_direction = rotation_matrix @ np.array([0, 0, -1])

    def signed_angle_xz(v1: np.ndarray, v2: np.ndarray) -> float:
        a1 = np.arctan2(v1[0], v1[2])
        a2 = np.arctan2(v2[0], v2[2])
        diff = a2 - a1
        while diff >  np.pi: diff -= 2 * np.pi
        while diff < -np.pi: diff += 2 * np.pi
        return diff

    cam_angles = []
    for cam_id in camera_views:
        if cam_id not in cameras:
            continue
        extr = np.array(cameras[cam_id]['extrinsics'], dtype=np.float32)
        R = extr[:, :3]
        cam_dir = R @ np.array([0, 0, 1])
        angle = signed_angle_xz(front_direction, cam_dir)
        cam_angles.append({'id': cam_id, 'angle': angle})

    cam_angles.sort(key=lambda x: x['angle'])
    front_idx = int(np.argmin([abs(c['angle']) for c in cam_angles]))
    # 循环移位，正面相机排第一
    sorted_cams = cam_angles[front_idx:] + cam_angles[:front_idx]
    return [c['id'] for c in sorted_cams]


def compute_c2w(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    由 world→cam 变换 [R|T] 计算 camera-to-world 矩阵 (4x4)。
    4DDress 约定：X_cam = R @ X_world + T
    """
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R.T
    c2w[:3, 3]  = -(R.T @ T)
    return c2w


def build_cameras_json(cameras: dict, sorted_views: list,
                        img_w: int, img_h: int) -> dict:
    """
    构建 cameras.json：包含所有视角的内参、外参和 c2w 矩阵。

    字段说明：
      role:        'source'（前视角）或 'target'（其余视角）
      sorted_idx:  排序后的位置（0=前, 1=左, 2=后, 3=右）
      intrinsics:  3x3 内参矩阵
      extrinsics:  [3,4] world→cam 外参矩阵
      c2w:         4x4 camera-to-world 矩阵（供 LHM 渲染器使用）
      img_wh:      [W, H]
    """
    cam_json = {}
    for i, cam_id in enumerate(sorted_views):
        if cam_id not in cameras:
            continue
        K    = np.array(cameras[cam_id]['intrinsics'], dtype=np.float32)
        extr = np.array(cameras[cam_id]['extrinsics'], dtype=np.float32)  # [3,4]
        R, T = extr[:, :3], extr[:, 3]
        c2w  = compute_c2w(R, T)
        cam_json[cam_id] = {
            'role':       'source' if i == 0 else 'target',
            'sorted_idx': i,
            'intrinsics': K.tolist(),
            'extrinsics': extr.tolist(),
            'c2w':        c2w.tolist(),
            'img_wh':     [img_w, img_h],
        }
    return cam_json


def get_image_size(item_dir: str, cam_id: str) -> tuple:
    """从实际图像读取分辨率，返回 (width, height)。"""
    img_dir = os.path.join(item_dir, 'Capture', cam_id, 'images')
    imgs = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    if not imgs:
        return 1080, 1080  # 4DDress 常见分辨率 fallback
    img = cv2.imread(imgs[0])
    if img is None:
        return 1080, 1080
    h, w = img.shape[:2]
    return w, h  # (W, H)


# ===========================================================================
# 2. SMPL-X 参数转换
# ===========================================================================

def load_smplx_pkl_raw(smplx_path: str) -> dict:
    """
    加载 4DDress SMPL-X pkl，返回原始参数（不做 PCA 解码）。

    参考 smpl_dataloader.py::SMPLXWapper._get_smpl_data_4ddress 的读取方式：
      - 4DDress pkl 中手部姿态为 12 个 PCA 系数，不是轴角（不可直接 reshape 为 (15,3)）
      - expression 为 10 维（4DDress 原始值），补零到 100 由 JSON 构建函数负责
      - 身体姿态（body_pose）为 63 维轴角，根节点旋转（global_orient）为 3 维轴角

    输出 key：
      global_orient:    (3,)   world 轴角，根节点
      body_pose:        (63,)  21 关节轴角（展平）
      betas:            (10,)
      transl:           (3,)
      left_hand_pose:   (12,)  原始 PCA 系数（需通过 SMPLXDecoder 解码为轴角）
      right_hand_pose:  (12,)  原始 PCA 系数
      jaw_pose:         (3,)
      leye_pose:        (3,)
      reye_pose:        (3,)
      expression:       (10,)  4DDress 原始 10 维
    """
    data = pickle.load(open(smplx_path, 'rb'))

    def _flat(key: str, n_fallback: int) -> np.ndarray:
        v = data.get(key, np.zeros(n_fallback, dtype=np.float32))
        return np.array(v, dtype=np.float32).flatten()

    return {
        'global_orient':    _flat('global_orient',   3)[:3],
        'body_pose':        _flat('body_pose',       63)[:63],  # 21×3，留展平形式
        'betas':            _flat('betas',           10)[:LHM_BETA_DIM],
        'transl':           _flat('transl',           3)[:3],
        'left_hand_pose':   _flat('left_hand_pose',  12),       # 12 PCA 系数
        'right_hand_pose':  _flat('right_hand_pose', 12),       # 12 PCA 系数
        'jaw_pose':         _flat('jaw_pose',         3)[:3],
        'leye_pose':        _flat('leye_pose',        3)[:3],
        'reye_pose':        _flat('reye_pose',        3)[:3],
        'expression':       _flat('expression',      10),       # 4DDress 原始 10 维
    }


class SMPLXDecoder:
    """
    将 4DDress raw pkl 参数（12 维 PCA 手部姿态）解码为全轴角表示。

    参考 smpl_dataloader.py::SMPLXWapper._get_smpl_data_4ddress 的逻辑：
      - 4DDress pkl 中 left_hand_pose/right_hand_pose 为 12 个 PCA 系数
      - 用 smplx.create(use_pca=True, num_pca_comps=12) 初始化，与 smpl_dataloader 的
        SMPLXLayer(num_pca_comps=12) 等效
      - 从 body.full_pose 按 SMPL-X 55 关节顺序切出各部位的全轴角

    使用 LHM 已配置好的 human_model_files（pretrained_models/human_model_files）,
    不需要额外指定模型路径。
    """

    # SMPL-X 55 个关节在 full_pose 中的切片（joint 索引）
    _SLICE = {
        'global_orient':    slice(0,  1),
        'body_pose':        slice(1,  22),
        'jaw_pose':         slice(22, 23),
        'leye_pose':        slice(23, 24),
        'reye_pose':        slice(24, 25),
        'left_hand_pose':   slice(25, 40),
        'right_hand_pose':  slice(40, 55),
    }

    def __init__(self, human_model_path: str, device: str = 'cpu'):
        if not _SMPLX_DECODER_AVAILABLE:
            raise ImportError('SMPLXDecoder 需要 torch 和 smplx 包（LHM 已包含）。')
        # use_pca=True, num_pca_comps=12 与 4DDress pkl 格式一致
        # 对应 smpl_dataloader.py SMPLXWapper: SMPLXLayer(num_pca_comps=12)
        self.body_model = smplx.create(
            human_model_path,
            model_type='smplx',
            gender='neutral',
            num_betas=10,
            num_expression_coeffs=10,
            use_pca=True,
            num_pca_comps=12,
            flat_hand_mean=True,
        ).to(device)
        self.device = device

    def decode(self, raw: dict) -> dict:
        """
        输入 load_smplx_pkl_raw 的输出，返回全轴角 dict：
          global_orient:    (3,)    世界轴角
          body_pose:        (21,3)  身体关节轴角
          jaw_pose:         (3,)
          leye_pose:        (3,)
          reye_pose:        (3,)
          left_hand_pose:   (15,3)  由 12 PCA 系数解码后的全轴角
          right_hand_pose:  (15,3)
          betas:            (10,)
          transl:           (3,)
          expression:       (10,)   4DDress 原始 10 维，补零到 100 留给 JSON 构建函数
        """
        def _t(v):
            return torch.from_numpy(
                np.array(v, dtype=np.float32).flatten()
            ).reshape(1, -1).to(self.device)

        with torch.no_grad():
            body = self.body_model(
                global_orient   = _t(raw['global_orient']),    # (1,3)
                body_pose       = _t(raw['body_pose']),        # (1,63)
                betas           = _t(raw['betas']),            # (1,10)
                transl          = _t(raw['transl']),           # (1,3)
                left_hand_pose  = _t(raw['left_hand_pose']),   # (1,12) PCA 系数
                right_hand_pose = _t(raw['right_hand_pose']),  # (1,12) PCA 系数
                jaw_pose        = _t(raw['jaw_pose']),         # (1,3)
                leye_pose       = _t(raw['leye_pose']),        # (1,3)
                reye_pose       = _t(raw['reye_pose']),        # (1,3)
                expression      = _t(raw['expression']),       # (1,10)
            )

        # full_pose: (1, 165) = (1, 55×3) 全关节轴角（手部 PCA 已解码）
        full = body.full_pose.squeeze(0).detach().cpu().numpy().reshape(55, 3)

        s = self._SLICE
        return {
            'global_orient':    full[s['global_orient']][0].copy(),  # (3,)
            'body_pose':        full[s['body_pose']].copy(),         # (21,3)
            'jaw_pose':         full[s['jaw_pose']][0].copy(),       # (3,)
            'leye_pose':        full[s['leye_pose']][0].copy(),      # (3,)
            'reye_pose':        full[s['reye_pose']][0].copy(),      # (3,)
            'left_hand_pose':   full[s['left_hand_pose']].copy(),    # (15,3)
            'right_hand_pose':  full[s['right_hand_pose']].copy(),   # (15,3)
            'betas':            np.array(raw['betas'],      dtype=np.float32),
            'transl':           np.array(raw['transl'],     dtype=np.float32),
            'expression':       np.array(raw['expression'], dtype=np.float32),
        }


def _pad_expression(expr_raw: np.ndarray) -> np.ndarray:
    """将任意长度的表情系数补零到 LHM_EXPR_DIM（100）维。"""
    out = np.zeros(LHM_EXPR_DIM, dtype=np.float32)
    n = min(len(expr_raw), LHM_EXPR_DIM)
    out[:n] = expr_raw[:n]
    return out


def convert_smplx_world_to_cam(smplx_world: dict,
                                R_cam: np.ndarray,
                                T_cam: np.ndarray) -> dict:
    """
    将 SMPL-X 参数从 4DDress 世界坐标系转换到指定相机坐标系。

    4DDress 外参约定：X_cam = R_cam @ X_world + T_cam

    转换说明：
      trans:        trans_cam = R_cam @ trans_world + T_cam
      global_orient: 旋转矩阵的变换：R_global_cam = R_cam @ R_global_world
                     其余关节姿态（body_pose、hand 等）为相对旋转，无需变换。
    """
    # --- 平移：世界坐标系 → 相机坐标系 ---
    trans_world = smplx_world['transl']
    trans_cam   = R_cam @ trans_world + T_cam

    # --- 根节点旋转：复合世界到相机的旋转 ---
    R_global_world = Rotation.from_rotvec(smplx_world['global_orient']).as_matrix()
    R_global_cam   = R_cam @ R_global_world
    root_pose_cam  = Rotation.from_matrix(R_global_cam).as_rotvec().astype(np.float32)

    return {
        'root_pose':        root_pose_cam,
        'body_pose':        smplx_world['body_pose'],        # 相对旋转，不变
        'betas':            smplx_world['betas'],
        'trans':            trans_cam.astype(np.float32),
        'lhand_pose':       smplx_world['left_hand_pose'],
        'rhand_pose':       smplx_world['right_hand_pose'],
        'jaw_pose':         smplx_world['jaw_pose'],
        'leye_pose':        smplx_world['leye_pose'],
        'reye_pose':        smplx_world['reye_pose'],
        'expr':             smplx_world['expression'],
    }


def build_lhm_smplx_json(smplx_cam: dict, K: np.ndarray,
                           img_w: int, img_h: int) -> dict:
    """
    构建与 video2motion.py::save_results 输出格式完全一致的 LHM SMPL-X JSON。

    字段格式参考（来自 video2motion.py save_results + prepare_motion_seqs）：
      root_pose:   list[3]        全局旋转 axis-angle
      body_pose:   list[21][3]    身体关节 axis-angle
      jaw_pose:    list[3]
      leye_pose:   list[3]
      reye_pose:   list[3]
      lhand_pose:  list[15][3]
      rhand_pose:  list[15][3]
      betas:       list[10]
      expr:        list[100]      （video2motion 原始输出无此字段，但 v2 格式有）
      trans:       list[3]        相机坐标系下根节点平移
      focal:       [fx, fy]       相机焦距（像素）
      princpt:     [cx, cy]       主点坐标（像素）
      img_size_wh: [W, H]         原始图像尺寸
    """
    def _ls(v) -> list:
        return np.array(v, dtype=np.float32).flatten().tolist()

    return {
        'root_pose':   _ls(smplx_cam['root_pose']),
        'body_pose':   _ls(smplx_cam['body_pose']),    # 63 值 (21×3)
        'jaw_pose':    _ls(smplx_cam['jaw_pose']),
        'leye_pose':   _ls(smplx_cam['leye_pose']),
        'reye_pose':   _ls(smplx_cam['reye_pose']),
        'lhand_pose':  _ls(smplx_cam['lhand_pose']),   # 45 值 (15×3 解码后轴角)
        'rhand_pose':  _ls(smplx_cam['rhand_pose']),   # 45 值
        'betas':       _ls(smplx_cam['betas']),
        'expr':        _ls(_pad_expression(np.array(smplx_cam['expr'], dtype=np.float32))),
        'trans':       _ls(smplx_cam['trans']),
        'focal':       [float(K[0, 0]), float(K[1, 1])],
        'princpt':     [float(K[0, 2]), float(K[1, 2])],
        'img_size_wh': [int(img_w), int(img_h)],
    }


# ===========================================================================
# 3. 帧索引解析
# ===========================================================================

def parse_frame_idx_from_pkl(pkl_path: str) -> tuple:
    """
    从 4DDress pkl 文件名解析帧索引。

    命名格式：mesh-{frame:06d}_smplx.pkl 或 mesh-{frame:06d}_smpl.pkl
    返回：(frame_idx_int, frame_idx_str)，如 (1, '000001')
    """
    fname = os.path.basename(pkl_path)  # 'mesh-000001_smplx.pkl'
    try:
        # split('-') → ['mesh', '000001_smplx.pkl']
        # split('_')[0] → '000001'
        frame_str = fname.split('-')[1].split('_')[0]
        return int(frame_str), frame_str
    except (IndexError, ValueError):
        raise ValueError(f"无法从文件名解析帧索引: {fname}")


# ===========================================================================
# 4. 单序列处理：SMPL-X 转换
# ===========================================================================

def process_sequence_smplx(
    item_dir: str,
    output_dir: str,
    sample_rate: int = 1,
    smplx_decoder: 'SMPLXDecoder | None' = None,
) -> bool:
    """
    处理单个 4DDress 序列的 SMPL-X 参数转换和相机参数保存。

    输出：
      smplx_params/{frame:05d}.json  — SMPL-X in 前视角相机坐标系
      cameras.json                   — 所有相机内外参（训练 Dataset 使用）

    Args:
        item_dir:      4DDress 序列路径 (.../subject/outfit/take)
        output_dir:    本序列输出目录
        sample_rate:   帧采样率（每 N 帧取一帧）
        smplx_decoder: SMPLXDecoder 实例（用于将 PCA 手部参数解码为轴角）；
                       为 None 时手部输出为零。

    Returns:
        True 表示成功，False 表示跳过/失败。
    """
    cameras_pkl = os.path.join(item_dir, 'Capture', 'cameras.pkl')
    if not os.path.exists(cameras_pkl):
        print(f"  [跳过] 找不到 cameras.pkl: {cameras_pkl}")
        return False

    cameras     = load_cameras_pkl(cameras_pkl)
    sorted_views = get_sorted_camera_views(item_dir, cameras)
    front_cam_id = sorted_views[0]

    img_w, img_h = get_image_size(item_dir, front_cam_id)
    front_extr   = np.array(cameras[front_cam_id]['extrinsics'], dtype=np.float32)
    R_front, T_front = front_extr[:, :3], front_extr[:, 3]
    K_front      = np.array(cameras[front_cam_id]['intrinsics'], dtype=np.float32)

    print(f"  前视角相机: {front_cam_id}，图像分辨率: {img_w}x{img_h}")
    print(f"  相机顺序（前→左→后→右）: {sorted_views}")

    # --- 确定 SMPL-X pkl 目录 ---
    smplx_dir = os.path.join(item_dir, 'SMPLX')
    smpl_dir  = os.path.join(item_dir, 'SMPL')
    if os.path.exists(smplx_dir):
        param_dir, suffix = smplx_dir, '_smplx.pkl'
    elif os.path.exists(smpl_dir):
        param_dir, suffix = smpl_dir, '_smpl.pkl'
        print("  [警告] 未找到 SMPLX，改用 SMPL（无手部/面部参数，相关字段置零）")
    else:
        print(f"  [跳过] 未找到 SMPL/SMPLX 目录: {item_dir}")
        return False

    all_pkl = sorted(glob.glob(os.path.join(param_dir, f'*{suffix}')))
    sampled_pkl = all_pkl[::sample_rate]
    if not sampled_pkl:
        print(f"  [跳过] pkl 文件为空: {param_dir}")
        return False

    # --- 创建输出目录 ---
    os.makedirs(os.path.join(output_dir, 'smplx_params'), exist_ok=True)

    if smplx_decoder is None:
        print("  [警告] 未提供 SMPLXDecoder，手部 PCA 姿态无法解码，"
              "lhand_pose/rhand_pose 将输出为零。")

    # --- 逐帧处理 ---
    n_success = 0
    for pkl_path in tqdm(sampled_pkl, desc='  SMPL-X 转换', leave=False):
        try:
            frame_idx, frame_str = parse_frame_idx_from_pkl(pkl_path)
        except ValueError as e:
            print(f"  [警告] {e}")
            continue

        try:
            # 按 smpl_dataloader.py 方式加载原始参数（手部为 PCA 系数）
            raw = load_smplx_pkl_raw(pkl_path)
        except Exception as e:
            print(f"  [警告] 加载 pkl 失败 {os.path.basename(pkl_path)}: {e}")
            continue

        # 解码 PCA 手部姿态 → 全轴角（参考 smpl_dataloader.py SMPLXWapper.decode）
        if smplx_decoder is not None:
            try:
                smplx_world = smplx_decoder.decode(raw)
            except Exception as e:
                print(f"  [警告] 解码失败 {os.path.basename(pkl_path)}: {e}")
                continue
        else:
            # 无解码器时退化：手部填零，其他字段直接用（body_pose 保留展平形式）
            smplx_world = {
                'global_orient':    raw['global_orient'],
                'body_pose':        raw['body_pose'].reshape(21, 3),
                'betas':            raw['betas'],
                'transl':           raw['transl'],
                'left_hand_pose':   np.zeros((15, 3), dtype=np.float32),
                'right_hand_pose':  np.zeros((15, 3), dtype=np.float32),
                'jaw_pose':         raw['jaw_pose'],
                'leye_pose':        raw['leye_pose'],
                'reye_pose':        raw['reye_pose'],
                'expression':       raw['expression'],
            }

        # 转换到前视角相机坐标系并保存
        smplx_cam = convert_smplx_world_to_cam(smplx_world, R_front, T_front)
        lhm_json  = build_lhm_smplx_json(smplx_cam, K_front, img_w, img_h)
        out_path  = os.path.join(output_dir, 'smplx_params', f'{frame_idx:05d}.json')
        with open(out_path, 'w') as f:
            json.dump(lhm_json, f)

        n_success += 1

    print(f"  SMPL-X 转换完成：{n_success}/{len(sampled_pkl)} 帧")

    # --- 保存 cameras.json ---
    cam_json      = build_cameras_json(cameras, sorted_views, img_w, img_h)
    cam_json_path = os.path.join(output_dir, 'cameras.json')
    with open(cam_json_path, 'w') as f:
        json.dump(cam_json, f, indent=2)
    print(f"  cameras.json 已保存（{len(cam_json)} 路相机）")

    return n_success > 0


# ===========================================================================
# 5. 单序列处理：FLAME 估计（需依赖 LHM_Track engines）
# ===========================================================================

def prepare_imgs_for_flame(
    item_dir: str,
    output_dir: str,
    front_cam_id: str,
    sample_rate: int = 1,
) -> int:
    """
    为 FLAME 估计在 imgs_png/ 下创建指向原始图像的软链接（不复制文件）。

    gaga_track 内部硬编码读取 {seq_dir}/imgs_png/*.png，并要求文件名为
    1-based 连续整数（{fidx+1:05d}.png），Sapiens/FLAME 输出以此为索引。

    Returns:
        准备的帧数
    """
    img_src_dir = os.path.join(item_dir, 'Capture', front_cam_id, 'images')
    img_files   = sorted(glob.glob(os.path.join(img_src_dir, '*.png')))
    sampled     = img_files[::sample_rate]

    imgs_png_dir = os.path.join(output_dir, 'imgs_png')
    os.makedirs(imgs_png_dir, exist_ok=True)

    for new_idx, src_path in enumerate(sampled, start=1):
        dst_path = os.path.join(imgs_png_dir, f'{new_idx:05d}.png')
        if not os.path.exists(dst_path):
            os.symlink(os.path.abspath(src_path), dst_path)

    return len(sampled)


def process_sequence_flame(
    output_dir: str,
    model_path: str,
    device: str = 'cuda',
) -> bool:
    """
    在已准备好 imgs_png/ 的序列目录上运行 FLAME 估计。

    流程：
      Step A: Sapiens 2D 关键点（run_sapiens）
              输入: imgs_png/
              输出: sapiens_pose/{fidx+1:05d}.json
      Step B: gaga_track FLAME 拟合（estimate_flame）
              依赖: imgs_png/ + sapiens_pose/
              输出: flame_params/{fidx+1:05d}.json

    注意：两步均需在 LHM_Track 目录下运行（sys.path 需包含 engine/）。
    """
    # 动态添加路径以访问 LHM_Track 的 engine 模块
    track_dir = os.path.dirname(os.path.abspath(__file__))
    engine_dir = os.path.join(track_dir, 'engine')
    for p in [track_dir, engine_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        from predict_sapiens_pose import run_sapiens
        from predict_flame import estimate_flame, init_gaga_track
    except ImportError as e:
        print(f"  [错误] 无法导入 LHM_Track 引擎模块: {e}")
        print("  请确保在 LHM_Track/ 目录下运行此脚本。")
        return False

    imgs_png_dir = os.path.join(output_dir, 'imgs_png')
    if not os.path.exists(imgs_png_dir) or not os.listdir(imgs_png_dir):
        print(f"  [跳过] imgs_png/ 不存在或为空: {imgs_png_dir}")
        return False

    # Step A: Sapiens 2D 关键点
    sapiens_out = os.path.join(output_dir, 'sapiens_pose')
    n_imgs      = len(glob.glob(os.path.join(imgs_png_dir, '*.png')))
    n_sapiens   = len(glob.glob(os.path.join(sapiens_out,  '*.json'))) if os.path.exists(sapiens_out) else 0

    if n_sapiens >= n_imgs:
        print(f"  Sapiens 关键点已存在（{n_sapiens} 帧），跳过")
    else:
        print(f"  运行 Sapiens 关键点估计（{n_imgs} 帧）...")
        run_sapiens(model_path, output_dir)

    # Step B: gaga_track FLAME 拟合
    flame_out = os.path.join(output_dir, 'flame_params')
    n_flame   = len(glob.glob(os.path.join(flame_out, '*.json'))) if os.path.exists(flame_out) else 0

    if n_flame >= n_imgs:
        print(f"  FLAME 参数已存在（{n_flame} 帧），跳过")
        return True

    print(f"  运行 FLAME 估计（需要 gagatracker 模型）...")
    try:
        gaga_model_path = os.path.join(model_path, 'gagatracker')
        gaga_track      = init_gaga_track(gaga_model_path, device)
        estimate_flame(gaga_track, output_dir)
        n_flame = len(glob.glob(os.path.join(flame_out, '*.json')))
        print(f"  FLAME 估计完成：{n_flame} 帧")
        return True
    except Exception:
        traceback.print_exc()
        print("  [错误] FLAME 估计失败")
        return False


# ===========================================================================
# 6. 数据集枚举与 meta JSON 生成
# ===========================================================================

def collect_sequences(
    root_dir: str,
    subjects: list = None,
    outfits:  list = None,
    takes:    list = None,
) -> list:
    """
    枚举 4DDress 数据集中所有有效序列，返回 seq_info list。

    每个 seq_info 字典：
      subject, outfit, take: 4DDress 层级名称
      item_dir:              完整路径
      uid:                   '{subject}_{outfit}_{take}'（用作训练 meta 中的标识符）
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"4DDress 根目录不存在: {root_dir}")

    all_subjects = subjects or sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and not d.endswith('.tar.gz')
    ])

    sequences = []
    for subj in all_subjects:
        subj_dir = os.path.join(root_dir, subj)
        for outfit in (outfits or ['Inner', 'Outer']):
            outfit_dir = os.path.join(subj_dir, outfit)
            if not os.path.isdir(outfit_dir):
                continue
            for take in sorted(os.listdir(outfit_dir)):
                if takes and take not in takes:
                    continue
                if 'DS_Store' in take:
                    continue
                take_dir = os.path.join(outfit_dir, take)
                if not os.path.isdir(take_dir):
                    continue
                # 最简校验：必须有 Capture/cameras.pkl
                if not os.path.exists(os.path.join(take_dir, 'Capture', 'cameras.pkl')):
                    continue
                sequences.append({
                    'subject':  subj,
                    'outfit':   outfit,
                    'take':     take,
                    'item_dir': take_dir,
                    'uid':      f'{subj}_{outfit}_{take}',
                })
    return sequences


def build_meta_json(output_dir: str, all_uids: list, val_ratio: float = 0.1):
    """
    生成训练/验证 meta JSON 文件。

    输出：
      {output_dir}/label/train_list.json  — UID 列表
      {output_dir}/label/val_list.json
    """
    split_idx  = max(1, int(len(all_uids) * (1 - val_ratio)))
    train_uids = all_uids[:split_idx]
    val_uids   = all_uids[split_idx:]

    label_dir = os.path.join(output_dir, 'label')
    os.makedirs(label_dir, exist_ok=True)

    with open(os.path.join(label_dir, 'train_list.json'), 'w') as f:
        json.dump(train_uids, f, indent=2)
    with open(os.path.join(label_dir, 'val_list.json'), 'w') as f:
        json.dump(val_uids, f, indent=2)

    print(f"\nMeta JSON 已生成：训练 {len(train_uids)} 条，验证 {len(val_uids)} 条")
    print(f"  路径：{label_dir}/")


# ===========================================================================
# 7. 单序列完整处理入口
# ===========================================================================

def process_single_sequence(seq_info: dict, output_dir: str, args,
                             smplx_decoder: 'SMPLXDecoder | None' = None) -> bool:
    """处理单个 4DDress 序列的完整流程。"""
    uid      = seq_info['uid']
    item_dir = seq_info['item_dir']
    seq_out  = os.path.join(output_dir, uid)
    os.makedirs(seq_out, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"序列：{uid}")
    print(f"  源路径：{item_dir}")
    print(f"  输出路径：{seq_out}")

    # ── SMPL-X 转换 ──────────────────────────────────────────────────────
    if args.mode in ('smplx', 'all'):
        ok = process_sequence_smplx(
            item_dir, seq_out,
            sample_rate   = args.sample_rate,
            smplx_decoder = smplx_decoder,
        )
        if not ok:
            print(f"  [跳过] SMPL-X 转换失败")
            return False

    # ── 图像准备 + FLAME 估计 ─────────────────────────────────────────────
    if args.mode in ('flame', 'all') and args.model_path:
        cameras_pkl  = os.path.join(item_dir, 'Capture', 'cameras.pkl')
        cameras      = load_cameras_pkl(cameras_pkl)
        sorted_views = get_sorted_camera_views(item_dir, cameras)
        front_cam_id = sorted_views[0]

        n_frames = prepare_imgs_for_flame(
            item_dir, seq_out, front_cam_id, args.sample_rate
        )
        print(f"  已准备 {n_frames} 帧图像用于 FLAME 估计")

        process_sequence_flame(seq_out, args.model_path, args.device)

    return True


# ===========================================================================
# 8. 验证工具
# ===========================================================================

def verify_sequence_output(output_dir: str, uid: str) -> dict:
    """
    检查单个序列的输出完整性，返回各类数据的帧数统计。
    """
    seq_dir  = os.path.join(output_dir, uid)
    result   = {'uid': uid, 'exists': os.path.isdir(seq_dir)}
    if not result['exists']:
        return result

    result['smplx_params']     = len(glob.glob(os.path.join(seq_dir, 'smplx_params', '*.json')))
    result['flame_params']     = len(glob.glob(os.path.join(seq_dir, 'flame_params', '*.json')))
    result['sapiens_pose']     = len(glob.glob(os.path.join(seq_dir, 'sapiens_pose', '*.json')))
    result['imgs_png']         = len(glob.glob(os.path.join(seq_dir, 'imgs_png',     '*.png')))
    result['has_cameras_json'] = os.path.exists(os.path.join(seq_dir, 'cameras.json'))

    return result


# ===========================================================================
# 9. 命令行入口
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='将 4DDress 数据集转换为 LHM 推理/训练格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--root_dir',    type=str, required=True,
                        help='4DDress 数据集根目录路径（服务器或本地）')
    parser.add_argument('--output_dir',  type=str, default='./train_data/4ddress_lhm',
                        help='输出目录（默认：./train_data/4ddress_lhm）')
    parser.add_argument('--model_path',  type=str, default='./pretrained_models',
                        help='LHM 预训练模型根目录（默认 ./pretrained_models）。'
                             'SMPL-X 模型从 {model_path}/human_model_files 加载，'
                             'FLAME 估计从同目录查找 gagatracker。')
    parser.add_argument('--mode', type=str,
                        choices=['smplx', 'flame', 'all'],
                        default='all',
                        help=(
                            'smplx: 只转换 SMPL-X 参数，生成 smplx_params/ 和 cameras.json; '
                            'flame: 只运行 FLAME 估计（需已有 imgs_png/）; '
                            'all:   执行全部步骤（默认）'
                        ))
    # 序列过滤
    parser.add_argument('--subject',     type=str, default=None,
                        help='只处理指定 subject（如 "00122"）')
    parser.add_argument('--outfit',      type=str, default=None,
                        help='只处理指定 outfit（"Inner" 或 "Outer"）')
    parser.add_argument('--take',        type=str, default=None,
                        help='只处理指定 take（如 "Take1"）')
    # 采样与输出控制
    parser.add_argument('--sample_rate', type=int, default=1,
                        help='帧采样率：每 N 帧取一帧（默认 1，取全部）')
    parser.add_argument('--build_meta',  action='store_true',
                        help='处理完成后生成 train_list.json / val_list.json')
    parser.add_argument('--val_ratio',   type=float, default=0.1,
                        help='验证集比例（默认 0.1）')
    parser.add_argument('--device',      type=str,  default='cuda',
                        help='FLAME 估计使用的设备（默认 cuda）')
    parser.add_argument('--verify',      action='store_true',
                        help='完成后打印每个序列的输出完整性报告')

    args = parser.parse_args()

    # ── 枚举序列 ─────────────────────────────────────────────────────────
    subjects = [args.subject] if args.subject else None
    outfits  = [args.outfit]  if args.outfit  else None
    takes    = [args.take]    if args.take    else None

    sequences = collect_sequences(args.root_dir, subjects, outfits, takes)
    print(f"共找到 {len(sequences)} 个 4DDress 序列待处理。")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 初始化 SMPLXDecoder（全局共享，避免每帧重复加载模型）──────────────
    # 使用 LHM 配置好的 human_model_files，与 smpl_dataloader.py 的读取逻辑保持一致
    smplx_decoder = None
    if args.mode in ('smplx', 'all'):
        human_model_path = os.path.join(args.model_path, 'human_model_files')
        if _SMPLX_DECODER_AVAILABLE and os.path.isdir(human_model_path):
            print(f"初始化 SMPLXDecoder (human_model_files: {human_model_path})")
            try:
                smplx_decoder = SMPLXDecoder(
                    human_model_path=human_model_path,
                    device=args.device,
                )
            except Exception as e:
                print(f"[警告] SMPLXDecoder 初始化失败: {e}")
                print("       手部 PCA 姿态无法解码，lhand/rhand 将输出为零。")
        else:
            if not _SMPLX_DECODER_AVAILABLE:
                print("[警告] 缺少 torch 或 smplx 包，SMPLXDecoder 不可用，手部输出为零。")
            else:
                print(f"[警告] human_model_files 不存在: {human_model_path}")
                print("       请确认 --model_path 指向正确的 pretrained_models 目录。")

    # ── 逐序列处理 ────────────────────────────────────────────────────────
    processed_uids = []
    failed_uids    = []

    for seq_info in tqdm(sequences, desc='处理序列', unit='seq'):
        try:
            ok = process_single_sequence(seq_info, args.output_dir, args,
                                          smplx_decoder=smplx_decoder)
            if ok:
                processed_uids.append(seq_info['uid'])
            else:
                failed_uids.append(seq_info['uid'])
        except Exception:
            traceback.print_exc()
            failed_uids.append(seq_info['uid'])
            print(f"  [错误] 序列处理异常: {seq_info['uid']}")

    print(f"\n{'='*60}")
    print(f"处理完成：{len(processed_uids)} 成功，{len(failed_uids)} 失败。")

    if failed_uids:
        print(f"失败序列：{failed_uids[:10]}{'...' if len(failed_uids) > 10 else ''}")

    # ── 生成 meta JSON ────────────────────────────────────────────────────
    if args.build_meta and processed_uids:
        build_meta_json(args.output_dir, processed_uids, args.val_ratio)

    # ── 验证输出 ──────────────────────────────────────────────────────────
    if args.verify and processed_uids:
        print('\n── 输出完整性验证 ──')
        print(f"{'UID':<40} {'smplx':>6} {'world':>6} {'flame':>6} {'sapiens':>8} {'imgs':>5} {'cam':>4}")
        print('─' * 80)
        for uid in processed_uids[:20]:  # 只显示前20条
            r = verify_sequence_output(args.output_dir, uid)
            cam_ok = '✓' if r.get('has_cameras_json') else '✗'
            print(f"{uid:<40} {r.get('smplx_params',0):>6} {r.get('smplx_params_world',0):>6} "
                  f"{r.get('flame_params',0):>6} {r.get('sapiens_pose',0):>8} "
                  f"{r.get('imgs_png',0):>5} {cam_ok:>4}")
        if len(processed_uids) > 20:
            print(f"  ... 还有 {len(processed_uids) - 20} 条未显示")


if __name__ == '__main__':
    main()
