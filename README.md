# BSAI SolarWM-H3 — ComfyUI 相机轨迹可控世界模型插件
# BSAI SolarWM-H3 — ComfyUI Camera-Trajectory World Model Plugin

[![BSAI](https://img.shields.io/badge/BSAI-Plugin%20%2316-purple)](https://github.com/xm6018924)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-0.30+-orange)](https://github.com/comfyanonymous/ComfyUI)
[![MiniMax-H3](https://img.shields.io/badge/MiniMax-H3-blue)](https://github.com/MiniMax-AI/MiniMax-H3)
[![SolarWM](https://img.shields.io/badge/SolarWM-Camera--Controlled-green)](https://github.com/Junchao-cs/SolarWM)

---

## 📖 项目简介 / Introduction

**中文**：
本插件将 [SolarWM](https://github.com/Junchao-cs/SolarWM) 的相机轨迹可控世界模型能力集成到 ComfyUI，基于 [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) 33B 多模态模型。通过 fused-PRoPE（融合投影旋转位置编码）技术，你可以用自然语言描述场景，并精确控制相机轨迹——环绕、推拉、摇摄、俯仰——生成电影级音视频。

**English**：
This plugin integrates [SolarWM](https://github.com/Junchao-cs/SolarWM)'s camera-trajectory world model into ComfyUI, built on the [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) 33B multimodal model. Using fused-PRoPE (Projective Rotary Positional Embedding), you describe a scene in natural language while precisely controlling the camera trajectory — orbit, dolly, pan, tilt — to generate cinematic video with synchronized audio.

### ✨ 核心特性 / Key Features

| 特性 Feature | 说明 Description |
|---|---|
| 🎥 **5 种相机轨迹 / 5 Camera Modes** | orbit环绕 / dolly推拉 / pan摇摄 / tilt俯仰 / static静止 |
| 🎯 **fused-PRoPE** | 4×4 相机位姿矩阵沿对角线重复构造 32×32 旋转矩阵，作用于 Attention |
| 🎛 **4 个独立节点 / 4 Nodes** | Loader / CameraTrajectory / ApplyCamera / Info |
| 🔊 **音视频同步 / AV Sync** | 视频 + 音频同时生成 |
| ⚡ **4步蒸馏 / 4-Step Distilled** | 支持 FastVideo 蒸馏模型，~5秒生成 |

---

## 📦 系统要求 / Requirements

| 项目 | 最低要求 |
|---|---|
| GPU VRAM | 24GB（推荐 32GB+） |
| ComfyUI | 0.30.0+（需原生支持 MiniMax-H3） |
| PyTorch | 2.5+（CUDA 12.x） |
| Python | 3.10+ |

---

## 🚀 安装教程 / Installation

### 步骤 1：克隆插件 / Step 1: Clone

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xm6018924/ComfyUI-BSAI-SolarWM-H3.git BSAI_ComfyUI_SolarWM_H3
```

### 步骤 2：下载模型 / Step 2: Download Models

你需要以下模型文件：

| 模型类型 | 文件名示例 | 下载来源 |
|---|---|---|
| 扩散模型 | `fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors` | ComfyUI-Manager / ModelScope |
| 全量模型 | `minimax_h3_fl2va_bf16.safetensors` | [MiniMax-H3 HF](https://huggingface.co/MiniMaxAI/MiniMax-H3) |
| 文本编码器 | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | ComfyUI-Manager |
| 视频 VAE | `minimax_h3_video_vae_fp16.safetensors` | ComfyUI-Manager |
| 音频 VAE | `minimax_h3_audio_vae_fp32.safetensors` | ComfyUI-Manager |
| **SolarWM LoRA** | `SolarWM-H3-stage2.pt` | [SolarWM-H3-33B HF](https://huggingface.co/junchaoh-cs/SolarWM-H3-33B) |

> ⚠️ **重要 / Important**：SolarWM LoRA 需要从 `SolarWM-h3-33B-sgf-stage2-158f/ema.pt` 下载，放到 `ComfyUI/models/loras/` 目录。

### 步骤 3：重启 ComfyUI / Step 3: Restart

重启后在节点列表中搜索 `BSAI SolarWM-H3`，确认 4 个节点加载成功。

---

## 🎛 节点说明 / Node Reference

### 1. BSAI SolarWM-H3 Loader（模型加载器）

加载 H3 模型 + SolarWM LoRA，并安装 fused-PRoPE 相机补丁。

| 参数 | 中文 | 说明 |
|---|---|---|
| `model_name` | 扩散模型 | 选择 H3 扩散模型文件 |
| `precision` | 精度 | int8 / default / fp8_e4m3fn |
| `solar_lora_name` | SolarWM LoRA | 选择 `SolarWM-H3-stage2.pt` |
| `solar_lora_strength` | LoRA 强度 | 默认 1.0 |
| `camera_prope_dim_start` | ProPE 起始维度 | 默认 96 |
| `camera_prope_dim_end` | ProPE 结束维度 | 默认 128 |
| `num_heads` | Attention 头数 | 33B 模型 = 40 |

### 2. BSAI SolarWM-H3 Camera Trajectory（相机轨迹生成）

生成 c2w 相机轨迹。

| 参数 | 中文 | 说明 | 推荐值 |
|---|---|---|---|
| `num_frames` | 总帧数 | 视频帧数（24fps） | 124（≈5秒） |
| `mode` | 相机模式 | orbit/dolly/pan/tilt/static | orbit |
| `orbit_angle` | 环绕角度 | 一圈=360° | 360 |
| `orbit_radius` | 环绕半径 | 相机离主体距离 | 5.0 |
| `orbit_height` | 相机高度 | 离地高度（米） | 1.5 |
| `start_angle` | 起始角度 | 环绕起始方向 | 0 |
| `dolly_speed` | 推拉速度 | 正=拉远，负=推近 | 0 |
| `pan_speed` | 摇摄速度 | 每帧水平摇镜头角度 | 0 |
| `tilt_speed` | 俯仰速度 | 每帧上下摇镜头角度 | 0 |

### 3. BSAI SolarWM-H3 Apply Camera（注入相机）

将相机轨迹注入模型。

| 参数 | 中文 | 说明 |
|---|---|---|
| `model` | 模型 | 连接 Loader 输出 |
| `camera_poses` | 相机位姿 | 连接 CameraTrajectory 输出 |
| `camera_seed` | 相机种子 | 随机种子（固定可复现） |

### 4. BSAI SolarWM-H3 Info（信息显示）

显示当前 SolarWM 配置状态。

---

## 🎬 示例工作流 / Example Workflow

参考 `workflows/SolarWM_H3_Basic.json`，基本连接如下：

```
Loader → ApplyCamera → KSampler → VAEDecode (视频) → VideoCombine
         ↑                  ↑
CameraTrajectory ──────────┘

CLIPLoader → CLIPTextEncode (正面/负面) → KSampler
VAELoader (视频) → VAEDecode
VAELoader (音频) → VAEDecodeAudio → VideoCombine
EmptyHunyuanLatentVideo → KSampler
```

### 推荐设置 / Recommended Settings

| 参数 | 蒸馏模型 / Distilled | 全量模型 / Full BF16 |
|---|---|---|
| 扩散模型 | `fastvideo_fasth3_8step_*` | `minimax_h3_fl2va_bf16` |
| 步数 / Steps | 4 | 50+ |
| CFG | 4.0 | 5.0 |
| 采样器 / Sampler | euler | euler_dpm |
| 分辨率 / Resolution | 1344×768 | 1344×768 |
| 帧数 / Frames | 124 | 124 |

---

## 🎨 提示词技巧 / Prompt Tips

**中文**：在正面提示词中描述场景、光照、风格，并明确提及相机运动（如 "camera smoothly orbits"）。负面提示词排除模糊、静态、无声等。

**English**：Describe the scene, lighting, and style in the positive prompt, and explicitly mention camera motion (e.g., "camera smoothly orbits around the subject"). Use negative prompts to exclude blur, static, silent, etc.

---

## 🔧 故障排除 / Troubleshooting

| 问题 | 解决方案 |
|---|---|
| LoRA 加载失败 | 确保 LoRA 文件名以 `.pt` 结尾，放在 `models/loras/` |
| 显存不足 | 使用 int8 蒸馏模型，减小分辨率或帧数 |
| 相机无效果 | 确认 ApplyCamera 节点连接正确，LoRA 已加载 |
| 视频闪烁 | 增加步数，或检查 LoRA 强度是否过高 |

---

## 📄 许可 / License

本插件仅供学习研究使用。SolarWM 权重遵循其原始许可协议。

This plugin is for research purposes. SolarWM weights follow their original license.

---

## 🙏 致谢 / Acknowledgments

- [SolarWM](https://github.com/Junchao-cs/SolarWM) — 相机可控世界模型
- [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) — 33B 多模态视频模型
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — 节点式 AI 工作流
- [BSAI](https://github.com/xm6018924) — 插件 #16

---

<div align="center">
BSAI Plugin #16 · SolarWM-H3 Camera Control for ComfyUI
</div>
