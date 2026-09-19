# -*- coding: utf-8 -*-
"""
BSAI ComfyUI SolarWM-H3 — ComfyUI 节点
=========================================
为 MiniMax H3 提供相机轨迹可控的世界模型推理 (SolarWM fused_prope)。

节点:
  1. BSAI SolarWM-H3 Camera Trajectory  — 生成相机 c2w 轨迹
  2. BSAI SolarWM-H3 Loader             — 加载 H3 + SolarWM LoRA + 安装相机补丁
  3. BSAI SolarWM-H3 Apply Camera      — 将相机轨迹注入模型
  4. BSAI SolarWM-H3 Info              — 显示状态
"""

import os
import math
import torch

try:
    import folder_paths
except Exception:
    folder_paths = None

from .camera import (
    generate_camera_trajectory,
    encode_camera_prope,
    FRAME_RESCALE,
)


# ============================================================================
# 工具函数
# ============================================================================

def _get_diffusion_models():
    try:
        if folder_paths:
            return folder_paths.get_filename_list("diffusion_models")
    except Exception:
        pass
    return ["model.safetensors"]


def _get_loras():
    try:
        if folder_paths:
            return folder_paths.get_filename_list("loras")
    except Exception:
        pass
    return []


# ============================================================================
# SolarWM 相机 ProPE Attention 补丁 (一次性安装)
# ============================================================================

_PATCH_INSTALLED = False


def _install_solarwm_attention_patch():
    """一次性 monkey-patch MiniMaxH3 Attention.forward, 注入相机 ProPE 嵌入。

    原理:
      - transformer_options["solarwm_cam_emb"] 存储 [N_cam, heads, cam_dim]
      - 在 Attention.forward 中, RoPE 之后、optimized_attention 之前,
        对视频 token 行将相机嵌入加到 q/k 的 [dim_start:dim_end] 维度
      - 无相机嵌入时走原始路径, 不影响正常 H3 推理
    """
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return
    _PATCH_INSTALLED = True

    try:
        from comfy.ldm.minimax.model import Attention as H3Attention
    except Exception:
        try:
            from comfy.ldm.minimax.model import Attention as H3Attention
        except Exception:
            print("[BSAI-SolarWM-H3] 警告: 无法导入 H3 Attention, 相机补丁未安装")
            return

    _orig_forward = H3Attention.forward

    def _patched_forward(self, x, rope_freqs=None, transformer_options=None):
        transformer_options = transformer_options or {}
        cam_emb = transformer_options.get("solarwm_cam_emb")

        # 无相机嵌入: 走原始路径
        if cam_emb is None:
            return _orig_forward(self, x, rope_freqs=rope_freqs,
                                 transformer_options=transformer_options)

        # 有相机嵌入: 执行原始 forward 后注入
        # 我们需要在 RoPE 之后、attention 之前修改 q/k
        # 由于原始 forward 内部完成全部计算, 我们重写关键路径
        s = x.shape[0]
        qkv = self.qkv_proj(x)
        head_dim = self.head_dim
        heads = self.heads
        q, k, v = qkv.split(heads * head_dim, dim=-1)
        v = v.view(s, heads, head_dim)

        if rope_freqs is not None:
            import comfy.model_management
            import comfy.quant_ops
            q = q.view(1, s, heads, head_dim)
            k = k.view(1, s, heads, head_dim)
            qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
            rot = rope_freqs.shape[-3] * 2
            if comfy.model_management.in_training:
                q, k = comfy.quant_ops.ck.rms_rope_split_half(
                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            else:
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            q = q[0]
            k = k[0]
        else:
            q = self.q_norm(q.view(s, heads, head_dim))
            k = self.k_norm(k.view(s, heads, head_dim))

        # === 相机 fused-PRoPE 旋转注入 ===
        cam_dim_start = transformer_options.get("solarwm_cam_dim_start", 96)
        cam_dim_end = transformer_options.get("solarwm_cam_dim_end", 128)
        cam_per_head = cam_dim_end - cam_dim_start  # 32
        cam_w2c = transformer_options.get("solarwm_cam_w2c")

        if cam_w2c is None or cam_per_head != 32:
            # 无相机矩阵, 走原始 attention
            pass
        else:
            layout = transformer_options.get("minimax_h3_layout")
            if layout is not None and hasattr(layout, "segments"):
                video_a = video_b = 0
                for a, b, kind in layout.segments:
                    if kind == "video":
                        video_a, video_b = a, b
                        break
                if video_b > video_a:
                    n_video = video_b - video_a
                    pos_ids = layout.position_ids
                    video_t = pos_ids[video_a:video_b, 0].to(q.device)
                    t_min = video_t[0].item(); t_max = video_t[-1].item()
                    t_range = t_max - t_min
                    n_cam = cam_w2c.shape[0]
                    if t_range > 1e-6:
                        norm_t = (video_t - t_min) / t_range
                    else:
                        norm_t = torch.zeros(n_video, device=q.device)
                    cam_idx = (norm_t * (n_cam - 1)).round().long().clamp(0, n_cam - 1)

                    # 构造 32x32 块对角旋转矩阵: w2c[4,4] 重复8次
                    # E_i = diag(w2c_i, w2c_i, ..., w2c_i)  (8 blocks of 4x4)
                    w2c_dev = cam_w2c.to(device=q.device, dtype=torch.float32)  # [N,4,4]
                    # 构造块对角矩阵 (下方循环)
                    # 正确: 构造 [N, 32, 32] 块对角矩阵
                    E_list = []
                    for i in range(n_cam):
                        Ei = torch.zeros(32, 32, device=q.device, dtype=torch.float32)
                        for blk in range(8):
                            Ei[blk*4:(blk+1)*4, blk*4:(blk+1)*4] = w2c_dev[i]
                        E_list.append(Ei)
                    E_all = torch.stack(E_list, dim=0)  # [N, 32, 32]
                    E_T = E_all.transpose(1, 2)  # [N, 32, 32]
                    E_inv = torch.linalg.inv(E_all)  # [N, 32, 32]

                    # 对 q 的相机维度施加 E_i^T
                    q_cam = q[video_a:video_b, :, cam_dim_start:cam_dim_end].float()  # [n_video, heads, 32]
                    q_cam = torch.einsum("nij,nhj->nhi", E_T[cam_idx], q_cam)  # [n_video, heads, 32]
                    q[video_a:video_b, :, cam_dim_start:cam_dim_end] = q_cam.to(q.dtype)

                    # 对 k 的相机维度施加 E_j^{-1}
                    k_cam = k[video_a:video_b, :, cam_dim_start:cam_dim_end].float()
                    k_cam = torch.einsum("nij,nhj->nhi", E_inv[cam_idx], k_cam)
                    k[video_a:video_b, :, cam_dim_start:cam_dim_end] = k_cam.to(k.dtype)

                    # 对 v 的相机维度施加 E_j^{-1}
                    v_cam = v[video_a:video_b, :, cam_dim_start:cam_dim_end].float()
                    v_cam = torch.einsum("nij,nhj->nhi", E_inv[cam_idx], v_cam)
                    v[video_a:video_b, :, cam_dim_start:cam_dim_end] = v_cam.to(v.dtype)

                    _E_out = E_all[cam_idx]  # attention 输出逆变换用

        # === 继续原始 attention ===
        from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
        q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
        k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
        v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
        out = optimized_attention(q, k, v, heads, mask=None, skip_reshape=True,
                                  transformer_options=transformer_options)
        out = out.squeeze(0)

        # attention 输出施加 E_i (逆变换回原始空间)
        if cam_w2c is not None and cam_per_head == 32 and "video_a" in locals():
            try:
                out_cam = out[video_a:video_b, :, cam_dim_start:cam_dim_end].float()
                out_cam = torch.einsum("nij,nhj->nhi", _E_out, out_cam)
                out[video_a:video_b, :, cam_dim_start:cam_dim_end] = out_cam.to(out.dtype)
            except Exception:
                pass

        return self.out_proj(out)

    H3Attention.forward = _patched_forward
    print("[BSAI-SolarWM-H3] 相机 ProPE Attention 补丁已安装 (fused_prope)")


# ============================================================================
# 节点 1: 相机轨迹生成
# ============================================================================

class BSAI_SolarWM_H3_CameraTrajectory:
    """生成相机 c2w 轨迹 (SolarWM 相机可控世界模型)。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_frames": ("INT", {
                    "default": 124, "min": 5, "max": 960, "step": 1,
                    "tooltip": "像素帧数 (24fps), H3 自动对齐到 17k+5 网格"
                }),
                "mode": (["orbit", "dolly", "pan", "tilt", "static"],
                         {"default": "orbit",
                          "tooltip": "orbit=环绕主体, dolly=推拉, pan=水平摇, tilt=垂直摇, static=静止"}),
                "orbit_angle": ("FLOAT", {
                    "default": 360.0, "min": -720.0, "max": 720.0, "step": 1.0,
                    "tooltip": "环绕总角度 (度), 360=一整圈"
                }),
                "orbit_radius": ("FLOAT", {
                    "default": 5.0, "min": 0.5, "max": 50.0, "step": 0.1,
                    "tooltip": "机位距离/环绕半径 (世界单位)"
                }),
                "orbit_height": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "相机高度 (世界单位)"
                }),
                "start_angle": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0,
                    "tooltip": "起始角度 (度)"
                }),
                "dolly_speed": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01,
                    "tooltip": "每帧推拉距离 (正=远离, 负=靠近/推近)"
                }),
                "pan_speed": ("FLOAT", {
                    "default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1,
                    "tooltip": "每帧水平摇镜头角度 (度)"
                }),
                "tilt_speed": ("FLOAT", {
                    "default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1,
                    "tooltip": "每帧垂直摇镜头角度 (度)"
                }),
            },
        }

    RETURN_TYPES = ("CAMERA_POSE",)
    RETURN_NAMES = ("camera_poses",)
    FUNCTION = "generate"
    CATEGORY = "BSAI/SolarWM-H3"
    DESCRIPTION = "生成 SolarWM-H3 相机 c2w 轨迹: orbit环绕/dolly推拉/pan摇/tilt俯仰"

    def generate(self, num_frames, mode, orbit_angle, orbit_radius, orbit_height,
                 start_angle, dolly_speed, pan_speed, tilt_speed):
        poses = generate_camera_trajectory(
            num_frames=num_frames,
            mode=mode,
            orbit_angle=orbit_angle,
            orbit_radius=orbit_radius,
            orbit_height=orbit_height,
            start_angle=start_angle,
            dolly_speed=dolly_speed,
            pan_speed=pan_speed,
            tilt_speed=tilt_speed,
        )
        print(f"[BSAI-SolarWM-H3] 相机轨迹生成: {mode}, {num_frames}帧, "
              f"角度={orbit_angle}°, 距离={orbit_radius}")
        return (poses,)


# ============================================================================
# 节点 2: SolarWM-H3 Loader
# ============================================================================

def _load_torch_file_safe(path):
    """安全加载 safetensors (兼容 header 数据长度不一致的文件)。"""
    try:
        import comfy.utils
        return comfy.utils.load_torch_file(path, safe_load=True)
    except Exception:
        pass
    import struct, json
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
        sd = {}
        _dtmap = {'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16,
                  'I8': torch.int8, 'I32': torch.int32, 'I64': torch.int64}
        for k, v in hdr.items():
            if k == '__metadata__':
                continue
            begin, end = v['data_offsets']
            f.seek(8 + n + begin)
            raw = f.read(end - begin)
            dt = _dtmap.get(v.get('dtype'), torch.float32)
            sd[k] = torch.frombuffer(raw, dtype=dt).reshape(v['shape'])
    return sd


def _solarwm_key_to_comfy(key):
    """SolarWM diffusers key -> ComfyUI MiniMax-H3 key."""
    k = key
    k = k.replace("base_model.model.", "")
    k = k.replace("transformer_blocks.", "blocks.")
    k = k.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
    k = k.replace(".attn.to_out.0", ".attn.out_proj")
    k = k.replace(".ff.net.0.proj", ".mlp.fc1")
    k = k.replace(".ff.net.2", ".mlp.fc2")
    return k


def _load_solarwm_lora(path):
    """Load SolarWM ema.pt / safetensors LoRA and convert to ComfyUI patches.
    
    Returns list of (target_key, B, A, rank) tuples ready for model.add_patches.
    """
    from comfy.weight_adapter.lora import LoRAAdapter
    
    # Try loading as PyTorch checkpoint first (ema.pt format)
    sd = None
    if path.endswith(".pt") or path.endswith(".pth"):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                if "shadow" in ckpt:
                    sd = ckpt["shadow"]
                elif "state_dict" in ckpt:
                    sd = ckpt["state_dict"]
                elif "model" in ckpt:
                    sd = ckpt["model"]
                else:
                    sd = ckpt
            print(f"[BSAI-SolarWM-H3] 从 PyTorch checkpoint 加载, keys={len(sd)}")
        except Exception as e:
            print(f"[BSAI-SolarWM-H3] PyTorch加载失败: {e}")
    
    if sd is None:
        try:
            sd = _load_torch_file_safe(path)
        except Exception as e:
            print(f"[BSAI-SolarWM-H3] safetensors加载失败: {e}")
            return {}
    
    # Collect lora_A/lora_B pairs
    ab = {}
    for k, v in sd.items():
        if k.endswith(".lora_A.weight"):
            base = k[:-len(".lora_A.weight")]
            bk = base + ".lora_B.weight"
            if bk in sd:
                ab[base] = (v, sd[bk])
    
    print(f"[BSAI-SolarWM-H3] LoRA pairs found: {len(ab)}")
    
    # QKV fusion: to_q/to_k/to_v -> qkv_proj block diagonal
    lora_patches = {}
    trio_groups = {}
    for base in list(ab.keys()):
        for role in (".attn.to_q", ".attn.to_k", ".attn.to_v"):
            if base.endswith(role):
                attn_prefix = base[:base.rfind(".")]
                trio_groups.setdefault(attn_prefix, {})[role.rsplit(".", 1)[-1]] = ab.pop(base)
                break
    
    for attn_prefix, trio in trio_groups.items():
        if all(r in trio for r in ("to_q", "to_k", "to_v")):
            A_q, B_q = trio["to_q"]
            A_k, B_k = trio["to_k"]
            A_v, B_v = trio["to_v"]
            r = A_q.shape[0]
            hidden = A_q.shape[1]
            inner = B_q.shape[0]
            dtype = A_q.dtype
            A = torch.zeros((3 * r, hidden), dtype=dtype)
            A[:r] = A_q; A[r:2*r] = A_k; A[2*r:] = A_v
            B = torch.zeros((3 * inner, 3 * r), dtype=B_q.dtype)
            B[:inner, :r] = B_q
            B[inner:2*inner, r:2*r] = B_k
            B[2*inner:, 2*r:] = B_v
            target = "diffusion_model." + _solarwm_key_to_comfy(attn_prefix) + ".qkv_proj.weight"
            lora_patches[target] = LoRAAdapter(None, (B, A, float(r), None, None, None))
        else:
            for role, val in trio.items():
                ab[attn_prefix + ".attn." + role] = val
    
    # Standard 1:1 LoRA
    for base, (A, B) in ab.items():
        target = "diffusion_model." + _solarwm_key_to_comfy(base) + ".weight"
        rank = A.shape[0]
        lora_patches[target] = LoRAAdapter(None, (B, A, float(rank), None, None, None))
    
    print(f"[BSAI-SolarWM-H3] Converted patches: {len(lora_patches)}")
    return lora_patches

class BSAI_SolarWM_H3_Loader:
    """加载 H3 模型 + SolarWM Stage2 LoRA + 安装相机 ProPE 补丁。"""

    @classmethod
    def INPUT_TYPES(cls):
        _loras = _get_loras()
        return {
            "required": {
                "model_name": (_get_diffusion_models(),),
                "precision": (["int8", "default", "fp8_e4m3fn"], {"default": "int8"}),
                "solar_lora_name": (
                    (_loras if _loras else ["SolarWM-H3-stage2.pt"]),
                    {"default": "SolarWM-H3-stage2.pt",
                     "tooltip": "SolarWM Stage2 SGF LoRA (rank 384, ~2B params)"}
                ),
                "solar_lora_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "SolarWM LoRA 强度"
                }),
                "camera_prope_dim_start": ("INT", {
                    "default": 96, "min": 0, "max": 128, "step": 1,
                    "tooltip": "相机嵌入在 head_dim 中的起始维度 (SolarWM 默认 96)"
                }),
                "camera_prope_dim_end": ("INT", {
                    "default": 128, "min": 1, "max": 128, "step": 1,
                    "tooltip": "相机嵌入在 head_dim 中的结束维度 (SolarWM 默认 128)"
                }),
                "num_heads": ("INT", {
                    "default": 40, "min": 8, "max": 80, "step": 1,
                    "tooltip": "H3 Attention head 数 (33B=40, 5120/128)"
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "solar_info")
    FUNCTION = "load"
    CATEGORY = "BSAI/SolarWM-H3"
    DESCRIPTION = "加载 H3 + SolarWM Stage2 LoRA + 安装 fused_prope 相机补丁"

    def load(self, model_name, precision, solar_lora_name, solar_lora_strength,
             camera_prope_dim_start, camera_prope_dim_end, num_heads):
        from comfy.sd import load_diffusion_model

        # 安装相机补丁 (一次性)
        _install_solarwm_attention_patch()

        model_path = folder_paths.get_full_path("diffusion_models", model_name) if folder_paths else model_name
        model = load_diffusion_model(model_path)

        # 设置 shift (与 Sol-H3 一致)
        try:
            ms = model.get_model_object("model_sampling")
            if hasattr(ms, 'set_parameters'):
                ms.set_parameters(shift=12.0, audio_shift=3.0)
            to = model.model_options.get("transformer_options", {})
            to["minimax_h3_sigma_shift_video"] = 12.0
            to["minimax_h3_sigma_shift_audio"] = 3.0
            model.model_options["transformer_options"] = to
        except Exception as e:
            print(f"[BSAI-SolarWM-H3] shift设置跳过: {e}")

        # 加载 SolarWM LoRA
        lora_state = "OFF"
        if solar_lora_name and solar_lora_name.strip() and solar_lora_name.strip().lower() != "none":
            lora_path = folder_paths.get_full_path("loras", solar_lora_name) if folder_paths else solar_lora_name
            if lora_path and os.path.exists(lora_path):
                try:
                    sd = _load_torch_file_safe(lora_path)
                    # SolarWM LoRA: 支持 ema.pt (diffusers格式) 和 safetensors
                    lora_patches = _load_solarwm_lora(lora_path)
                    if lora_patches:
                        n = len(model.add_patches(lora_patches, strength_patch=solar_lora_strength, strength_model=1.0))
                        lora_state = f"{solar_lora_name} x{solar_lora_strength:.2f}"
                        print(f"[BSAI-SolarWM-H3] SolarWM LoRA 已加载: {solar_lora_name} "
                              f"strength={solar_lora_strength:.2f} ({n}/{len(lora_patches)} patches)")
                    else:
                        print(f"[BSAI-SolarWM-H3] 警告: LoRA {solar_lora_name} 无匹配 key")
                except Exception as e:
                    print(f"[BSAI-SolarWM-H3] LoRA 加载失败: {e}")
            else:
                print(f"[BSAI-SolarWM-H3] LoRA 文件不存在, 跳过: {solar_lora_name}")

        # 存储相机配置到 model
        model.model_options["transformer_options"]["solarwm_prope_config"] = {
            "dim_start": camera_prope_dim_start,
            "dim_end": camera_prope_dim_end,
            "num_heads": num_heads,
        }

        info = (f"SolarWM-H3: {model_name} | LoRA={lora_state} | "
                f"ProPE=[{camera_prope_dim_start}:{camera_prope_dim_end}] | "
                f"heads={num_heads}")
        print(f"[BSAI-SolarWM-H3] {info}")

        return (model, info)


# ============================================================================
# 节点 3: Apply Camera (将相机轨迹注入模型)
# ============================================================================

class BSAI_SolarWM_H3_ApplyCamera:
    """将相机 c2w 轨迹编码为 ProPE 嵌入并注入模型。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "camera_poses": ("CAMERA_POSE",),
                "camera_seed": ("INT", {
                    "default": 42, "min": 0, "max": 9999,
                    "tooltip": "ProPE 投影矩阵种子 (改变相机嵌入分布)"
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "BSAI/SolarWM-H3"
    DESCRIPTION = "将相机 c2w 轨迹编码为 fused_prope 嵌入并注入 H3 Attention"

    def apply(self, model, camera_poses, camera_seed):
        # 从 model 获取相机配置
        to = model.model_options.get("transformer_options", {})
        config = to.get("solarwm_prope_config", {})
        dim_start = config.get("dim_start", 96)
        dim_end = config.get("dim_end", 128)
        num_heads = config.get("num_heads", 40)

        cam_per_head = dim_end - dim_start

        # 编码相机轨迹 -> ProPE 嵌入
        cam_emb = encode_camera_prope(
            camera_poses,
            camera_prope_dim=cam_per_head,
            num_heads=num_heads,
            seed=camera_seed,
        )

        print(f"[BSAI-SolarWM-H3] 相机轨迹已注入: {camera_poses.shape[0]}帧, "
              f"嵌入={tuple(cam_emb.shape)}, ProPE=[{dim_start}:{dim_end}]")

        # 存储到 transformer_options (采样时由 Attention 补丁读取)
        to["solarwm_cam_emb"] = cam_emb
        to["solarwm_cam_dim_start"] = dim_start
        to["solarwm_cam_dim_end"] = dim_end
        model.model_options["transformer_options"] = to

        return (model,)


# ============================================================================
# 节点 4: Info 显示
# ============================================================================

class BSAI_SolarWM_H3_Info:
    """显示 SolarWM-H3 状态信息。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"solar_info": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ()
    FUNCTION = "show"
    CATEGORY = "BSAI/SolarWM-H3"
    OUTPUT_NODE = True

    def show(self, solar_info):
        return {"ui": {"text": [solar_info]}}


# ============================================================================
# 节点映射
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "BSAI_SolarWM_H3_CameraTrajectory": BSAI_SolarWM_H3_CameraTrajectory,
    "BSAI_SolarWM_H3_Loader": BSAI_SolarWM_H3_Loader,
    "BSAI_SolarWM_H3_ApplyCamera": BSAI_SolarWM_H3_ApplyCamera,
    "BSAI_SolarWM_H3_Info": BSAI_SolarWM_H3_Info,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_SolarWM_H3_CameraTrajectory": "BSAI SolarWM-H3 Camera Trajectory (相机轨迹)",
    "BSAI_SolarWM_H3_Loader": "BSAI SolarWM-H3 Loader (加载+相机补丁)",
    "BSAI_SolarWM_H3_ApplyCamera": "BSAI SolarWM-H3 Apply Camera (注入相机)",
    "BSAI_SolarWM_H3_Info": "BSAI SolarWM-H3 Info",
}
