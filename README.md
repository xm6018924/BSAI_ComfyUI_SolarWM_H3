# BSAI SolarWM-H3 / BSAI SolarWM-H3

**MiniMax H3 Camera-Trajectory Video Generation via SolarWM fused-PRoPE injection**

**通过 SolarWM fused-PRoPE 注入实现 MiniMax H3 相机轨迹可控视频生成**

---

## Features / 功能特性

- **One-click Generate node / 一键生成节点**: Merges camera PRoPE attach + 4-step distill sampling into a single node. / 将相机 PRoPE 注入与4步蒸馏采样合并为单个节点。
- **Full camera control / 完整相机控制**: Orbit, dolly, height sweep, pan, tilt, start angle, look-at offset. / 环绕、推拉、高度扫动、水平摇摄、俯仰、起始角度、注视点偏移。
- **Slider bars / 滑块条**: Drag sliders left (negative) / right (positive) to adjust values in real-time. / 拖动滑块左负右正，实时调节数值。
- **Keyboard arrows / 键盘方向键**: Focus a parameter, press up/down for step, left/right for fine adjustment. / 聚焦参数后，上下键步进，左右键微调。
- **Mouse wheel / 鼠标滚轮**: Scroll on any numeric field to adjust. / 在数值上滚动即可调节。
- **4-step distill / 4步蒸馏**: Works with MiniMax H3 turbo 4-step LoRA for fast generation. / 配合 MiniMax H3 turbo 4步蒸馏 LoRA 快速生成。
- **Bilingual tooltips / 双语提示**: Hover over any parameter for Chinese-English tooltip. / 鼠标悬停参数显示中英双语说明。

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

### Method 1: ComfyUI Manager / 方法一：ComfyUI 管理器

Search for BSAI_SolarWM_H3 in ComfyUI Manager and click Install. / 在 ComfyUI 管理器中搜索 BSAI_SolarWM_H3 并点击安装。

### Method 2: Git Clone / 方法二：Git 克隆

```bash
cd ComfyUI/custom_nodes
git clone git@github.com:xm6018924/BSAI_ComfyUI_SolarWM_H3.git
```

Restart ComfyUI. / 重启 ComfyUI。

---

## Nodes / 节点

### BSAI_SolarWM_H3_Generate (One-click) / BSAI_SolarWM_H3_Generate（一键生成）

Combines camera PRoPE injection + RandomNoise + BasicGuider + KSamplerSelect + BasicScheduler + SamplerCustomAdvanced. / 合并相机 PRoPE 注入 + 随机噪声 + 基础引导器 + 采样器选择 + 调度器 + 高级采样。

**Inputs / 输入:**

| Input / 输入 | Type / 类型 | Description / 说明 |
|---|---|---|
| model | MODEL | H3 diffusion model (after SolarWM LoRA) / H3 扩散模型（已加载 SolarWM LoRA） |
| positive | CONDITIONING | Positive prompt conditioning / 正向提示词条件 |
| latent | LATENT | H3 AV latent (from MiniMaxH3ImageToVideo) / H3 音视频潜空间 |

**Parameters / 参数:**

| Parameter / 参数 | Type / 类型 | Default / 默认 | Range / 范围 | Description / 说明 |
|---|---|---|---|---|
| orbit_turns | FLOAT | 0.0 | -1.0 ~ 1.0 | Orbit turns, negative=reverse / 环绕圈数（负=反向） |
| radius | FLOAT | 3.0 | 0.1 ~ 100.0 | Start camera distance / 起始相机距离 |
| radius_end | FLOAT | 3.0 | 0.1 ~ 100.0 | End distance (dolly-in/out) / 结束距离（推近/拉远） |
| height | FLOAT | 0.0 | -10.0 ~ 10.0 | Camera height / 相机高度 |
| height_end | FLOAT | 0.0 | -10.0 ~ 10.0 | End height (rise/fall) / 结束高度（上升/下降） |
| start_angle | FLOAT | 0.0 | 0.0 ~ 1.0 | Start angle (0=front, 0.25=side) / 起始角度（0=前方，0.25=侧方） |
| pan_speed | FLOAT | 0.0 | -1.0 ~ 1.0 | Horizontal pan speed / 水平摇摄速度 |
| tilt_speed | FLOAT | 0.0 | -3.14 ~ 3.14 | Vertical tilt speed (radians) / 俯仰速度（弧度） |
| look_at_y | FLOAT | 0.0 | -10.0 ~ 10.0 | Look-at point Y offset / 注视点高度 |
| seed | INT | 42 | 0 ~ 2^64 | Random seed / 随机种子 |
| steps | INT | 4 | 1 ~ 100 | Denoise steps (4 for distill) / 去噪步数（蒸馏用4） |
| sampler_name | COMBO | res_multistep | - | Sampler selection / 采样器选择 |
| scheduler | COMBO | simple | - | Scheduler selection / 调度器选择 |

**Output / 输出:**

| Output / 输出 | Type / 类型 | Description / 说明 |
|---|---|---|
| latent | LATENT | Sampled H3 AV latent / 采样后的 H3 音视频潜空间 |

### BSAI_SolarWM_H3_CameraAttach (Legacy) / BSAI_SolarWM_H3_CameraAttach（兼容）

Original camera-only node, outputs a patched MODEL. / 原始仅相机节点，输出打补丁后的 MODEL。

---

## Example Workflow / 示例工作流

An example workflow is included at: / 示例工作流位于：

```
workflows/BSAI_SolarWM_H3_Example.json
```

**Workflow structure / 工作流结构:**

```
UNETLoader -> LoraLoaderModelOnly(turbo) -> LoraLoaderModelOnly(solarwm) --+
CLIPLoader -----------------------------------------------------------------|
VAELoader(video) -----------------------------------------------------------|
                                                                           v
                                          MiniMaxH3ImageToVideo -> positive + latent
                                                                           |
                                          BSAI_SolarWM_H3_Generate <--------+
                                                  |
                                          +-------+-------+
                                          v               v
                                    VAEDecode      VAEDecodeAudio
                                          |               |
                                          +-------+-------+
                                                  v
                                          CreateVideo -> SaveVideo
```

**Quick start / 快速开始:**

1. Load BSAI_SolarWM_H3_Example.json in ComfyUI. / 在 ComfyUI 中加载示例工作流。
2. Set your reference image and prompt in MiniMaxH3ImageToVideo. / 设置参考图和提示词。
3. Adjust camera parameters on the Generate node. / 调节相机参数。
4. Queue prompt. / 提交运行。

---

## Camera Parameter Guide / 相机参数指南

| Motion / 运动 | Parameters / 参数 | Example / 示例 |
|---|---|---|
| Orbit around subject / 环绕主体 | orbit_turns=0.05 | Slow clockwise orbit / 慢速顺时针环绕 |
| Dolly in / 推近 | radius=3, radius_end=1.5 | Camera moves closer / 相机向前推近 |
| Dolly out / 拉远 | radius=2, radius_end=5 | Camera pulls back / 相机向后拉远 |
| Crane up / 升镜头 | height=0, height_end=2 | Camera rises upward / 相机上升 |
| Pan left / 左摇 | pan_speed=0.1 | Horizontal pan left / 水平左摇 |
| Tilt up / 上俯仰 | tilt_speed=0.3 | Camera tilts up / 相机上仰 |
| High angle / 俯拍 | height=3, look_at_y=0.5 | Looking down / 俯视 |
| Low angle / 仰拍 | height=-1.5, look_at_y=0.5 | Looking up / 仰视 |

---

## Changelog / 更新日志

### v2.1.0
- Added enhanced camera builder: height, height_end, start_angle, pan_speed, tilt_speed, look_at_y / 新增完整相机参数
- Added slider bars for all FLOAT parameters / 所有数值参数添加滑块条
- Added keyboard arrow key support / 添加键盘方向键支持
- Bilingual tooltips for all parameters / 全部参数双语提示
- Cleaned up deprecated files / 清理旧文件

### v2.0.0
- Complete rewrite based on verified SolarWM implementation / 基于验证通过的实现完整重写
- One-click Generate node (camera + sampler merged) / 一键生成节点
- Fused-PRoPE injection via model.clone + add_object_patch / fused-PRoPE 注入

---

## License / 许可

MIT License

---

## Author / 作者

BSAI (xm6018924)
