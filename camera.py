# -*- coding: utf-8 -*-
"""
BSAI ComfyUI SolarWM-H3 — 相机轨迹生成与 ProPE 编码
=====================================================
基于 SolarWM (fused_prope / logd4 / wan_fixed intrinsics) 的相机条件模块。

相机轨迹:
  - Orbit(环绕): 相机围绕主体旋转
  - Dolly(推拉): 相机沿光轴前后移动
  - Pan(平移):   相机水平旋转
  - Tilt(俯仰):  相机垂直旋转
  - Static(静止): 固定机位

ProPE 编码 (Wan2.1 world model 风格):
  c2w [4,4] -> rotation R[3,3] -> axis-angle [3]
            -> translation t[3] -> log(|t|+4) (logd4)
  合成 6D 相机位姿编码 [6]，投影到 [num_heads, camera_dim] 注入 Attention。
"""

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 相机基础数学
# ---------------------------------------------------------------------------

def _look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """计算 look-at 方向的 c2w 矩阵 (OpenGL 约定: 相机看向 -Z)。

    返回 [4, 4] float32, 列为主:
      c2w[:, 0] = right
      c2w[:, 1] = up
      c2w[:, 2] = -forward
      c2w[:, 3] = position
    """
    eye = torch.as_tensor(eye, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    up = torch.as_tensor(up, dtype=torch.float64)

    forward = target - eye
    forward = forward / (forward.norm() + 1e-8)
    right = torch.cross(forward, up, dim=0)
    right = right / (right.norm() + 1e-8)
    up_corrected = torch.cross(right, forward, dim=0)

    c2w = torch.eye(4, dtype=torch.float64)
    c2w[0, 0] = right[0]
    c2w[1, 0] = right[1]
    c2w[2, 0] = right[2]
    c2w[0, 1] = up_corrected[0]
    c2w[1, 1] = up_corrected[1]
    c2w[2, 1] = up_corrected[2]
    c2w[0, 2] = -forward[0]
    c2w[1, 2] = -forward[1]
    c2w[2, 2] = -forward[2]
    c2w[0, 3] = eye[0]
    c2w[1, 3] = eye[1]
    c2w[2, 3] = eye[2]
    return c2w.float()


def _rotation_matrix(axis, angle):
    """Rodrigues 公式: 绕单位轴旋转 angle 弧度。返回 [3, 3]。"""
    axis = torch.as_tensor(axis, dtype=torch.float64)
    axis = axis / (axis.norm() + 1e-8)
    K = torch.tensor([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ], dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
    return R.float()


def _apply_rotation_to_c2w(c2w, R_extra):
    """在相机坐标系上叠加旋转 (绕相机自身坐标轴)。"""
    R_extra = R_extra.double()
    rot_4x4 = torch.eye(4, dtype=torch.float64)
    rot_4x4[:3, :3] = R_extra
    return (c2w.double() @ rot_4x4).float()


# ---------------------------------------------------------------------------
# 相机轨迹生成
# ---------------------------------------------------------------------------

def generate_camera_trajectory(
    num_frames: int,
    mode: str = "orbit",
    orbit_angle: float = 360.0,
    orbit_radius: float = 5.0,
    orbit_height: float = 1.5,
    start_angle: float = 0.0,
    dolly_speed: float = 0.0,
    pan_speed: float = 0.0,
    tilt_speed: float = 0.0,
    target=(0.0, 1.0, 0.0),
):
    """生成 num_frames 帧的 c2w 相机轨迹。

    参数:
      num_frames: 像素帧数 (24fps)
      mode: orbit / dolly / pan / tilt / static
      orbit_angle: 环绕总角度 (度)
      orbit_radius: 环绕半径 / 初始机位距离
      orbit_height: 环绕高度
      start_angle: 起始角度 (度)
      dolly_speed: 每帧推拉距离 (正=远离, 负=靠近)
      pan_speed: 每帧水平摇镜头角度 (度)
      tilt_speed: 每帧垂直摇镜头角度 (度)
      target: 注视点 (世界坐标)

    返回:
      poses: [num_frames, 4, 4] float32 c2w 矩阵
    """
    poses = []
    total_angle_rad = math.radians(orbit_angle)
    start_rad = math.radians(start_angle)

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)  # 0..1

        if mode == "orbit":
            angle = start_rad + total_angle_rad * t
            radius = orbit_radius + dolly_speed * i
            eye = (
                radius * math.cos(angle),
                orbit_height,
                radius * math.sin(angle),
            )
            c2w = _look_at(eye, target)

        elif mode == "dolly":
            # 沿光轴前后移动 (固定注视方向, 沿初始 forward 方向)
            angle = start_rad
            radius = orbit_radius + dolly_speed * i
            eye = (
                radius * math.cos(angle),
                orbit_height,
                radius * math.sin(angle),
            )
            c2w = _look_at(eye, target)

        elif mode == "pan":
            # 水平摇镜头: 固定机位, 旋转朝向
            angle = start_rad
            eye = (
                orbit_radius * math.cos(angle),
                orbit_height,
                orbit_radius * math.sin(angle),
            )
            c2w = _look_at(eye, target)
            pan_rad = math.radians(pan_speed * i)
            R_pan = _rotation_matrix((0.0, 1.0, 0.0), pan_rad)
            c2w = _apply_rotation_to_c2w(c2w, R_pan)

        elif mode == "tilt":
            angle = start_rad
            eye = (
                orbit_radius * math.cos(angle),
                orbit_height,
                orbit_radius * math.sin(angle),
            )
            c2w = _look_at(eye, target)
            tilt_rad = math.radians(tilt_speed * i)
            # 倾斜绕相机右轴
            R_tilt = _rotation_matrix((1.0, 0.0, 0.0), tilt_rad)
            c2w = _apply_rotation_to_c2w(c2w, R_tilt)

        else:  # static
            angle = start_rad
            eye = (
                orbit_radius * math.cos(angle),
                orbit_height,
                orbit_radius * math.sin(angle),
            )
            c2w = _look_at(eye, target)

        poses.append(c2w)

    return torch.stack(poses, dim=0)  # [N, 4, 4]


# ---------------------------------------------------------------------------
# ProPE 相机编码 (fused_prope / logd4 / wan_fixed)
# ---------------------------------------------------------------------------

def c2w_to_axis_angle(R):
    """旋转矩阵 -> 轴角表示 [3] (angle * axis)。

    使用 trace 法: theta = arccos((tr(R)-1)/2)
    axis = (R21-R12, R02-R20, R10-R01) / (2*sin(theta))
    """
    R = R.double()
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    theta = math.acos(cos_theta)

    if abs(theta) < 1e-8:
        return torch.zeros(3, dtype=torch.float64)

    if abs(theta - math.pi) < 1e-6:
        # theta ~ pi: 需要特殊处理
        # 找 R 中最大的对角元素
        diag = R.diagonal()
        k = int(torch.argmax(diag).item())
        axis = torch.zeros(3, dtype=torch.float64)
        # 公式: axis = sqrt((R[i,i]+1)/2) 符号从其他元素推断
        axis[k] = 1.0
        return (axis * theta).float()

    sin_theta = math.sin(theta)
    rx = (R[2, 1] - R[1, 2]) / (2.0 * sin_theta)
    ry = (R[0, 2] - R[2, 0]) / (2.0 * sin_theta)
    rz = (R[1, 0] - R[0, 1]) / (2.0 * sin_theta)
    axis = torch.tensor([rx, ry, rz], dtype=torch.float64)
    axis = axis / (axis.norm() + 1e-8)
    return (axis * theta).float()


def encode_camera_prope(c2w_poses, camera_prope_dim=32, num_heads=40, seed=42):
    """将 c2w 序列编码为 ProPE 相机嵌入。

    流程:
      1. c2w [N,4,4] -> R[N,3,3], t[N,3] (世界坐标相机位置)
      2. R -> axis-angle [N,3]
      3. t -> logd4: sign(t)*log(|t|+4) [N,3]
      4. 拼接 6D 编码 [N,6]
      5. 固定随机投影 6 -> [num_heads, camera_prope_dim]

    参数:
      c2w_poses: [N, 4, 4] float32
      camera_prope_dim: 每个 head 中相机嵌入占的维度数 (SolarWM 默认 32, head_dim=128)
      num_heads: Attention head 数
      seed: 投影矩阵随机种子 (确定性)

    返回:
      cam_emb: [N, num_heads, camera_prope_dim] float32
    """
    N = c2w_poses.shape[0]

    # 1) 提取 R 和 t
    R = c2w_poses[:, :3, :3].double()  # [N,3,3]
    t = c2w_poses[:, :3, 3].double()   # [N,3]

    # 2) axis-angle
    axis_angles = []
    for i in range(N):
        aa = c2w_to_axis_angle(R[i])
        axis_angles.append(aa)
    axis_angles = torch.stack(axis_angles, dim=0).double()  # [N,3]

    # 3) logd4 translation
    t_enc = torch.sign(t) * torch.log(torch.abs(t) + 4.0)  # [N,3]

    # 4) 拼接 6D
    cam_6d = torch.cat([axis_angles, t_enc], dim=-1).float()  # [N,6]

    # 5) 固定随机投影 6 -> [num_heads * camera_prope_dim]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    proj = torch.randn(6, num_heads * camera_prope_dim, generator=gen) / math.sqrt(6)
    # 归一化投影
    proj = proj / (proj.norm(dim=0, keepdim=True) + 1e-8)

    emb = cam_6d @ proj  # [N, num_heads * camera_prope_dim]
    emb = emb.view(N, num_heads, camera_prope_dim)

    return emb.float()


# ---------------------------------------------------------------------------
# 像素帧 -> latent 帧映射 (用于将相机嵌入分配到视频 token)
# ---------------------------------------------------------------------------

# H3 FRAME_PER_TOKEN (从 comfy.ldm.minimax.model 导入)
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0


def pixel_frame_to_latent_frame(pixel_frame_idx, latent_t):
    """像素帧索引 -> 对应的 latent 帧索引。

    H3 每个 latent 帧 k 覆盖的像素帧数为 FRAME_PER_TOKEN[k % 5],
    时间轴以 FRAME_RESCALE 为单位累积。

    简化: 均匀映射 — latent 帧 k 覆盖像素帧
      [k * 24 * FRAME_RESCALE / latent_t, ...]
    这里用线性近似。
    """
    # 每个 latent 帧覆盖的像素帧数 (平均)
    # total pixel frames = latent_t * FRAME_RESCALE * avg_FRAME_PER_TOKEN
    # 实际 FRAME_PER_TOKEN = (1,4,4,4,4), avg = 3.4
    # 但 pixel_frame 是 24fps 的, latent 时间轴是 FRAME_RESCALE=5/3 单位
    # 近似: latent_frame = pixel_frame / (avg_frames_per_latent)
    # avg frames per latent token = sum(FRAME_PER_TOKEN)/5 * FRAME_RESCALE?
    # 简化为线性映射
    return int(pixel_frame_idx * latent_t / max(1, pixel_frame_idx + 1))


def build_camera_token_map(latent_t, frame_rows, num_pixel_frames):
    """构建 [num_video_tokens] 的相机帧索引映射。

    每个视频 token 对应一个 latent 帧, 同一 latent 帧内的所有空间 token
    共享同一相机位姿。

    返回:
      token_cam_idx: [latent_t * frame_rows] long, 每个 token 对应的相机帧索引 (0..num_pixel_frames-1)
    """
    total_tokens = latent_t * frame_rows
    token_cam_idx = torch.zeros(total_tokens, dtype=torch.long)

    # 每个 latent 帧对应的像素帧 (线性插值)
    for k in range(latent_t):
        # latent 帧 k 的中心像素帧
        center_pixel = int((k + 0.5) * num_pixel_frames / latent_t)
        center_pixel = min(center_pixel, num_pixel_frames - 1)
        start = k * frame_rows
        end = start + frame_rows
        token_cam_idx[start:end] = center_pixel

    return token_cam_idx
