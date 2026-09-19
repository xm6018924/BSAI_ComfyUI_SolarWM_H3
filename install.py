# -*- coding: utf-8 -*-
"""
BSAI ComfyUI SolarWM-H3 自动安装脚本
======================================
首次安装/更新时自动:
1. 检查 PyTorch + CUDA 环境
2. 提示下载 SolarWM Stage2 LoRA
3. 验证 H3 原生支持
"""
import os
import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
COMFY_ROOT = SCRIPT_DIR.parent.parent  # custom_nodes/.. = ComfyUI/
LORA_DIR = COMFY_ROOT / "models" / "loras"
PYTHON = sys.executable


def log(msg):
    print(f"[BSAI-SolarWM-H3] {msg}", flush=True)


def check_h3_support():
    """检查 ComfyUI 是否支持 MiniMax H3。"""
    try:
        sys.path.insert(0, str(COMFY_ROOT))
        from comfy.ldm.minimax.model import MiniMaxH3Model
        log("MiniMax H3 模型支持已就绪 ✅")
        return True
    except Exception as e:
        log(f"MiniMax H3 支持检查: {e}")
        log("请升级 ComfyUI 到 0.30.0+ (H3 Day-0 支持)")
        return False


def check_torch():
    """检查 PyTorch CUDA 可用性。"""
    try:
        import torch
        log(f"PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}, "
            f"GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        return True
    except Exception as e:
        log(f"PyTorch 检查失败: {e}")
        return False


def print_lora_guide():
    """打印 SolarWM LoRA 下载指南。"""
    log("=" * 60)
    log("SolarWM-H3 Stage2 LoRA 下载指南:")
    log("-" * 60)
    log("1. 访问 HuggingFace:")
    log("   https://huggingface.co/junchaoh-cs/SolarWM-H3-33B")
    log("2. 下载 SolarWM-h3-33B-sgf-stage2-158f/ 目录")
    log("3. 将 LoRA 权重文件放到:")
    log(f"   {LORA_DIR}")
    log("4. 在 Loader 节点中选择该 LoRA 文件")
    log("-" * 60)
    log("注: 无 LoRA 时插件仍可加载 H3 基础模型,")
    log("    但相机轨迹控制效果需要 SolarWM Stage2 LoRA。")
    log("=" * 60)


def main():
    log("=" * 60)
    log("BSAI ComfyUI SolarWM-H3 安装检查")
    log("=" * 60)

    check_torch()
    check_h3_support()
    print_lora_guide()

    log("安装检查完成! 重启 ComfyUI 后使用。")
    log(f"工作流位置: {SCRIPT_DIR / 'workflows'}")


if __name__ == "__main__":
    main()
