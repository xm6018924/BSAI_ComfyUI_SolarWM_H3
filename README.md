# BSAI SolarWM-H3 / MiniMax H3 可控相机轨迹视频生成插件

**MiniMax H3 Camera-Trajectory Video Generation via SolarWM fused-PRoPE injection**

**通过 SolarWM fused-PRoPE 注入实现 MiniMax H3 相机轨迹可控视频生成**

---

## Features / 功能特性

- **One-click pipeline / 一键工作流**: Loader + Generate in two nodes. / 两个节点完成全部流程。
- **Combined model loader / 合并模型加载器**: Base model + turbo LoRA + SolarWM LoRA in one node. / 基础模型 + 蒸馏 LoRA + 相机 LoRA 合并为一个节点。
- **Full camera control / 完整相机控制**: Orbit, dolly, height sweep, pan, tilt, start angle, look-at offset. / 环绕、推拉、高度扫动、水平摇摄、俯仰、起始角度、注视点偏移。
- **Keyboard arrows / 键盘方向键**: Focus a parameter, press up/down for step, left/right for fine. / 聚焦参数后上下步进、左右微调。
- **Mouse wheel / 鼠标滚轮**: Scroll on numeric fields to adjust. / 在数值上滚动调节。
- **4-step distill / 4步蒸馏**: Fast generation with H3 turbo LoRA. / 配合 H3 turbo LoRA 快速生成。
- **Bilingual tooltips / 双语提示**: Hover for Chinese-English tooltip. / 悬停显示中英双语说明。

---

## Requirements / 依赖

- ComfyUI 0.36.0+
- MiniMax H3 base model
- SolarWM LoRA (solarwm_h3_stage05_adapter_model_comfy_new.safetensors)
- Turbo 4-step LoRA (minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors)
- Qwen3VL text encoder
- H3 Video VAE + Audio VAE

---

## Installation / 安装

```bash
cd ComfyUI/custom_nodes
git clone git@github.com:xm6018924/BSAI_ComfyUI_SolarWM_H3.git
```

Restart ComfyUI. / 重启 ComfyUI。

---

## Nodes / 节点

### 1. BSAI_SolarWM_H3_Loader (模型加载)

Combines UNETLoader + turbo LoraLoaderModelOnly + SolarWM LoraLoaderModelOnly into one node. / 将基础模型加载 + 蒸馏 LoRA + 相机 LoRA 三个节点合并为一个。

**Parameters / 参数:**

| Parameter / 参数 | Type / 类型 | Description / 说明 |
|---|---|---|
| base_model | COMBO | H3 diffusion model from models/diffusion_models / H3 扩散模型 |
| turbo_lora | COMBO | 4-step distill LoRA from models/loras / 4步蒸馏 LoRA |
| turbo_strength | FLOAT | Turbo LoRA strength (default 1.0) / 蒸馏 LoRA 强度 |
| solarwm_lora | COMBO | SolarWM camera adapter LoRA from models/loras / 相机适配器 LoRA |
| solarwm_strength | FLOAT | SolarWM LoRA strength (default 1.0) / 相机 LoRA 强度 |

**Output / 输出:**

| Output / 输出 | Type / 类型 | Description / 说明 |
|---|---|---|
| model | MODEL | Ready-to-use H3 model with both LoRAs applied / 已加载双 LoRA 的 H3 模型 |

---

### 2. BSAI_SolarWM_H3_Generate (一键生成)

Combines camera PRoPE injection + RandomNoise + BasicGuider + KSamplerSelect + BasicScheduler + SamplerCustomAdvanced. / 合并相机注入与全部采样节点。

**Inputs / 输入:**

| Input / 输入 | Type / 类型 | Description / 说明 |
|---|---|---|
| model | MODEL | From BSAI_SolarWM_H3_Loader / 来自模型加载节点 |
| positive | CONDITIONING | Positive prompt / 正向提示词 |
| latent | LATENT | H3 AV latent from MiniMaxH3ImageToVideo / 音视频潜空间 |

**Camera Parameters / 相机参数:**

| Parameter / 参数 | Type / 类型 | Default / 默认 | Range / 范围 | Description / 说明 |
|---|---|---|---|---|
| orbit_turns | FLOAT | 0.0 | -1.0 ~ 1.0 | Orbit turns, negative=reverse / 环绕圈数（负=反向） |
| radius | FLOAT | 3.0 | 0.1 ~ 100.0 | Start camera distance / 起始距离 |
| radius_end | FLOAT | 3.0 | 0.1 ~ 100.0 | End distance (dolly-in/out) / 结束距离 |
| height | FLOAT | 0.0 | -10 ~ 10 | Camera height / 相机高度 |
| height_end | FLOAT | 0.0 | -10 ~ 10 | End height (rise/fall) / 结束高度 |
| start_angle | FLOAT | 0.0 | 0 ~ 1 | Start angle (0=front, 0.25=side) / 起始角度 |
| pan_speed | FLOAT | 0.0 | -1 ~ 1 | Horizontal pan speed / 水平摇摄 |
| tilt_speed | FLOAT | 0.0 | -3.14 ~ 3.14 | Vertical tilt (radians) / 俯仰弧度 |
| look_at_y | FLOAT | 0.0 | -10 ~ 10 | Look-at point Y / 注视点高度 |

**Sampling Parameters / 采样参数:**

| Parameter / 参数 | Type / 类型 | Default / 默认 | Description / 说明 |
|---|---|---|---|
| seed | INT | 42 | Random seed / 随机种子 |
| steps | INT | 4 | Denoise steps (4 for distill) / 去噪步数 |
| sampler_name | COMBO | res_multistep | Sampler / 采样器 |
| scheduler | COMBO | simple | Scheduler / 调度器 |

**Output / 输出:**

| Output / 输出 | Type / 类型 | Description / 说明 |
|---|---|---|
| latent | LATENT | Sampled H3 AV latent / 采样后潜空间 |

---

### 3. BSAI_SolarWM_H3_CameraAttach (相机轨迹 - 兼容)

Legacy camera-only node, outputs patched MODEL. / 兼容节点，仅相机注入。

---

## Example Workflow / 示例工作流

```
workflows/BSAI_SolarWM_H3_Example.json
```

**Workflow structure / 工作流结构:**

```
BSAI_SolarWM_H3_Loader ──model──┐
                                  ▼
CLIPLoader ──┐                   BSAI_SolarWM_H3_Generate
VAELoader ────┤                   │           ▲
              ▼                   │           │
      MiniMaxH3ImageToVideo ──────┴───────────┘
                  │
                  ▼
         +────────┴────────+
         ▼                 ▼
    VAEDecode        VAEDecodeAudio
         │                 │
         +────────┬────────+
                  ▼
           CreateVideo -> SaveVideo
```

---

## Camera Parameter Guide / 相机参数指南

| Motion / 运动 | Parameters / 参数 | Example / 示例 |
|---|---|---|
| Orbit / 环绕 | orbit_turns=0.05 | Slow clockwise / 慢速顺时针 |
| Dolly in / 推近 | radius=3, radius_end=1.5 | Move closer / 向前推近 |
| Dolly out / 拉远 | radius=2, radius_end=5 | Pull back / 向后拉远 |
| Crane up / 升镜 | height=0, height_end=2 | Rises upward / 相机上升 |
| Pan / 摇摄 | pan_speed=0.1 | Horizontal pan / 水平摇 |
| Tilt / 俯仰 | tilt_speed=0.3 | Vertical tilt / 垂直俯仰 |
| High angle / 俯拍 | height=3, look_at_y=0.5 | Looking down / 俯视 |
| Low angle / 仰拍 | height=-1.5, look_at_y=0.5 | Looking up / 仰视 |

---

## Changelog / 更新日志

### v2.2.0
- Added BSAI_SolarWM_H3_Loader: combines UNETLoader + turbo LoRA + SolarWM LoRA / 合并模型加载器
- Fixed widget ordering and kwarg names / 修复控件排序和参数名
- Removed deprecated broken nodes / 删除废弃节点

### v2.1.0
- Added enhanced camera: height, pan, tilt, start_angle, look_at / 新增完整相机参数
- Added keyboard arrow key support / 键盘方向键
- Bilingual tooltips / 双语提示

### v2.0.0
- Complete rewrite based on verified SolarWM implementation / 完整重写
- One-click Generate node / 一键生成节点
- Fused-PRoPE injection / fused-PRoPE 注入

---

## License / 许可

MIT License

## Author / 作者

BSAI (xm6018924)
