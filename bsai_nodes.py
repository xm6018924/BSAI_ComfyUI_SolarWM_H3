"""BSAI SolarWM-H3 Camera Attach node (bilingual).

Adapts the working SolarWMCameraAttach with bilingual Chinese-English labels.
The core PRoPE math lives in payload.py and inject.py.
"""
from __future__ import annotations

from .inject import attach_solarwm
from .payload import (
    SolarWMCamera,
    SolarWMProPE,
    build_orbit_camera,
    build_row_plan,
    latent_pixel_count,
)


def _conditioning_text_len(positive):
    """Text token count from a ComfyUI CONDITIONING input."""
    try:
        context = positive[0][0]
    except Exception:
        return None
    if context is None:
        return None
    shape = tuple(getattr(context, "shape", ()))
    if len(shape) == 3:
        return int(shape[1])
    if len(shape) == 2:
        return int(shape[0])
    return None


def _latent_video_dims(latent):
    """Best-effort (latent_t, latent_h, latent_w) of the H3 video latent."""
    try:
        samples = latent.get("samples") if isinstance(latent, dict) else None
        if samples is None:
            return None
        if hasattr(samples, "is_nested") and getattr(samples, "is_nested", False):
            video = samples.tensors[0]
        elif isinstance(samples, (tuple, list)):
            video = samples[0]
        else:
            video = samples
        shape = tuple(getattr(video, "shape", ()))
        if len(shape) != 5:
            return None
        return int(shape[2]), int(shape[3]), int(shape[4])
    except Exception:
        return None


class BSAI_SolarWM_H3_CameraAttach:
    """BSAI SolarWM-H3 相机轨迹注入 / Camera Trajectory Inject.

    Clones the H3 model and attaches a camera-aware PRoPE path.
    Bypass this node to keep the model stock (A/B reference).

    Parameters / 参数:
      orbit_turns_环绕圈数: 围绕Y轴旋转圈数 (负=反向), e.g. -0.02=微推
      radius_起始半径:      起始距离 (世界单位), start distance
      radius_end_结束半径:   结束距离 (线性推拉), end distance for dolly
      latent_视频潜空间:     H3 AV LATENT input (required for frame count)
      positive_正向条件:     Positive CONDITIONING (required for text_len)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_模型": ("MODEL",),
                "orbit_turns_环绕圈数": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "环绕圈数 (负=反向) | Orbit turns (negative=reverse), e.g. -0.02=slow dolly-in test",
                }),
                "radius_起始半径": ("FLOAT", {
                    "default": 3.0, "min": 0.1, "max": 100.0, "step": 0.05,
                    "tooltip": "起始相机距离 | Start camera distance",
                }),
                "radius_end_结束半径": ("FLOAT", {
                    "default": 3.0, "min": 0.1, "max": 100.0, "step": 0.05,
                    "tooltip": "结束距离 (r_end<r=推近, r_end>r=拉远) | End distance (dolly-in/out)",
                }),
                "latent_视频潜空间": ("LATENT",),
                "positive_正向条件": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model_模型",)
    FUNCTION = "attach"
    CATEGORY = "BSAI/SolarWM-H3"
    DESCRIPTION = "BSAI SolarWM-H3 相机轨迹注入 (fused-PRoPE) | Camera trajectory inject"

    def attach(self, model_模型, orbit_turns_环绕圈数, radius_起始半径,
               radius_end_结束半径, latent_视频潜空间, positive_正向条件):
        video = _latent_video_dims(latent_视频潜空间)
        if video is None:
            raise ValueError(
                "BSAI SolarWM-H3: 需要 H3 AV LATENT (samples [B,24,T,H,W])"
            )
        latent_t, latent_h, latent_w = video
        traj_frames = latent_pixel_count(latent_t)

        text_len = _conditioning_text_len(positive_正向条件)
        if text_len is None:
            raise ValueError(
                "BSAI SolarWM-H3: 需要 positive CONDITIONING 来获取文本长度"
            )

        prope = SolarWMProPE(
            camera=SolarWMCamera(
                c2w=build_orbit_camera(
                    traj_frames,
                    turn=float(orbit_turns_环绕圈数),
                    radius=float(radius_起始半径),
                    radius_end=float(radius_end_结束半径),
                )
            )
        )

        row_plan = build_row_plan(
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            trajectory_frames=prope.camera.frames,
            text_len=text_len,
        )
        if row_plan.warning:
            print(f"[BSAI-SolarWM-H3] [row plan] {row_plan.warning}")
        print(
            "[BSAI-SolarWM-H3] row plan: "
            f"latent_t={row_plan.latent_t} "
            f"canvas={row_plan.latent_h}x{row_plan.latent_w} "
            f"frame_rows={row_plan.frame_rows} "
            f"video_rows={row_plan.video_rows} "
            f"text_len={row_plan.text_len} "
            f"traj={row_plan.trajectory_frames}"
        )

        patcher = attach_solarwm(model_模型, prope=prope, row_plan=row_plan)
        return (patcher,)


NODE_CLASS_MAPPINGS = {
    "BSAI_SolarWM_H3_CameraAttach": BSAI_SolarWM_H3_CameraAttach,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_SolarWM_H3_CameraAttach": "BSAI SolarWM-H3 Camera Attach (相机轨迹)",
}
