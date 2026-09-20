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
                "model": ("MODEL",),
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
                "latent": ("LATENT",),
                "positive": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "attach"
    CATEGORY = "BSAI/SolarWM-H3"
    DESCRIPTION = "BSAI SolarWM-H3 相机轨迹注入 (fused-PRoPE) | Camera trajectory inject"

    def attach(self, model, orbit_turns_环绕圈数, radius_起始半径,
               radius_end_结束半径, latent, positive):
        video = _latent_video_dims(latent)
        if video is None:
            raise ValueError(
                "BSAI SolarWM-H3: 需要 H3 AV LATENT (samples [B,24,T,H,W])"
            )
        latent_t, latent_h, latent_w = video
        traj_frames = latent_pixel_count(latent_t)

        text_len = _conditioning_text_len(positive)
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

        patcher = attach_solarwm(model, prope=prope, row_plan=row_plan)
        return (patcher,)


class BSAI_SolarWM_H3_Generate:
    """BSAI SolarWM-H3 One-click Generate (PRoPE camera + 4-step distill sample)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "orbit_turns": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "环绕圈数(负=反向) | Orbit turns (negative=reverse)",
                }),
                "radius": ("FLOAT", {
                    "default": 3.0, "min": 0.1, "max": 100.0, "step": 0.05,
                    "tooltip": "起始相机距离 | Start camera distance",
                }),
                "radius_end": ("FLOAT", {
                    "default": 3.0, "min": 0.1, "max": 100.0, "step": 0.05,
                    "tooltip": "结束距离(推近/拉远) | End distance (dolly-in/out)",
                }),
                "height": ("FLOAT", {
                    "default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05,
                    "tooltip": "相机高度 | Camera height (Y axis)",
                }),
                "height_end": ("FLOAT", {
                    "default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05,
                    "tooltip": "结束高度(上升/下降) | End height (rise/fall)",
                }),
                "start_angle": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "起始角度(0=前方,0.25=侧方) | Start angle offset",
                }),
                "pan_speed": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "水平摇摄速度 | Horizontal pan speed",
                }),
                "tilt_speed": ("FLOAT", {
                    "default": 0.0, "min": -3.14, "max": 3.14, "step": 0.05,
                    "tooltip": "俯仰速度(弧度) | Vertical tilt speed (rad)",
                }),
                "look_at_y": ("FLOAT", {
                    "default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05,
                    "tooltip": "注视点高度 | Look-at point Y offset",
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "随机种子 | Random seed",
                }),
                "steps": ("INT", {
                    "default": 4, "min": 1, "max": 100, "step": 1,
                    "tooltip": "去噪步数(4步蒸馏用4) | Denoise steps",
                }),
                "sampler_name": (["res_multistep", "euler", "euler_cfg_pp",
                                    "dpm_2", "dpm_2_ancestral", "heun", "heunpp2",
                                    "euler_ancestral", "euler_ancestral_cfg_pp"], {
                    "default": "res_multistep",
                    "tooltip": "采样器 | Sampler",
                }),
                "scheduler": (["simple", "sgm_uniform", "karras",
                                "exponential", "normal", "ddim_uniform"], {
                    "default": "simple",
                    "tooltip": "调度器 | Scheduler",
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"
    CATEGORY = "BSAI/SolarWM-H3"

    def generate(self, model, positive, latent,
                 orbit_turns, radius, radius_end, height, height_end,
                 start_angle, pan_speed, tilt_speed, look_at_y,
                 seed, steps, sampler_name, scheduler):
        import comfy.samplers
        import comfy.sample
        import latent_preview
        from comfy_extras.nodes_custom_sampler import Guider_Basic

        video = _latent_video_dims(latent)
        if video is None:
            raise ValueError("BSAI SolarWM-H3: need H3 AV LATENT")
        latent_t, latent_h, latent_w = video
        traj_frames = latent_pixel_count(latent_t)

        text_len = _conditioning_text_len(positive)
        if text_len is None:
            raise ValueError("BSAI SolarWM-H3: need positive CONDITIONING")

        from .payload import build_enhanced_camera
        c2w = build_enhanced_camera(
            traj_frames,
            orbit_turns=float(orbit_turns),
            radius=float(radius),
            radius_end=float(radius_end),
            height=float(height),
            height_end=float(height_end),
            start_angle=float(start_angle),
            pan_speed=float(pan_speed),
            tilt_speed=float(tilt_speed),
            look_at_y=float(look_at_y),
        )
        prope = SolarWMProPE(
            camera=SolarWMCamera(c2w=c2w)
        )
        row_plan = build_row_plan(
            latent_t=latent_t, latent_h=latent_h, latent_w=latent_w,
            trajectory_frames=prope.camera.frames, text_len=text_len,
        )
        if row_plan.warning:
            print(f"[BSAI-SolarWM-H3] [row plan] {row_plan.warning}")
        print(f"[BSAI-SolarWM-H3] row plan: t={row_plan.latent_t} "
              f"canvas={row_plan.latent_h}x{row_plan.latent_w} "
              f"text_len={row_plan.text_len} traj={row_plan.trajectory_frames}")

        model_prope = attach_solarwm(model, prope=prope, row_plan=row_plan)

        model_sampling = model.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(
            model_sampling, scheduler, steps
        ).cpu()

        guider = Guider_Basic(model_prope)
        guider.set_conds(positive)
        sampler = comfy.samplers.sampler_object(sampler_name)

        latent_out = latent.copy()
        latent_image = latent_out["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(
            model_prope, latent_image,
            latent_out.get("downscale_ratio_spacial", None),
            latent_out.get("downscale_ratio_temporal", None),
        )
        latent_out["samples"] = latent_image

        noise_mask = latent_out.get("noise_mask", None)
        x0_output = {}
        callback = latent_preview.prepare_callback(
            model_prope, sigmas.shape[-1] - 1, x0_output
        )
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        batch_inds = latent_out.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        samples = guider.sample(
            noise, latent_image, sampler, sigmas,
            denoise_mask=noise_mask, callback=callback,
            disable_pbar=disable_pbar, seed=seed,
        )
        samples = samples.to(comfy.model_management.intermediate_device())

        latent_out.pop("downscale_ratio_spacial", None)
        latent_out.pop("downscale_ratio_temporal", None)
        latent_out["samples"] = samples
        return (latent_out,)

NODE_CLASS_MAPPINGS = {
    "BSAI_SolarWM_H3_CameraAttach": BSAI_SolarWM_H3_CameraAttach,
    "BSAI_SolarWM_H3_Generate": BSAI_SolarWM_H3_Generate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_SolarWM_H3_CameraAttach": "BSAI SolarWM-H3 Camera Attach (相机轨迹)",
    "BSAI_SolarWM_H3_Generate": "BSAI SolarWM-H3 Generate (一键生成)",
}


# ============================================================================
# Backward-compatibility nodes (for v2-003 and earlier workflows)
# These let old workflows load without "missing node" errors.
# New workflows should use BSAI_SolarWM_H3_CameraAttach.
# ============================================================================

class BSAI_SolarWM_H3_Loader:
    """[DEPRECATED] Use standard CheckpointLoader + our CameraAttach instead."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name_模型": ([""], {"default": ""}),
                "lora_name_LoRA": ([""], {"default": ""}),
                "lora_strength_LoRA强度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model_模型", "solar_info_信息")
    FUNCTION = "load"
    CATEGORY = "BSAI/SolarWM-H3 (Deprecated)"
    DESCRIPTION = "[DEPRECATED] Old loader — use standard model loader + CameraAttach"

    def load(self, model_name_模型, lora_name_LoRA, lora_strength_LoRA强度):
        raise RuntimeError(
            "[BSAI-SolarWM-H3] BSAI_SolarWM_H3_Loader is deprecated.\n"
            "Please use the standard CheckpointLoader/UNETLoader + our "
            "BSAI_SolarWM_H3_CameraAttach node instead.\n"
            "See README.md for the updated workflow."
        )


class BSAI_SolarWM_H3_CameraTrajectory:
    """[DEPRECATED] Old camera trajectory generator — use CameraAttach."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "总帧数_num_frames": ("INT", {"default": 124, "min": 5, "max": 960, "step": 1}),
                "相机模式_mode": (["orbit", "dolly", "pan", "tilt", "static"], {"default": "orbit"}),
                "环绕角度_orbit_angle": ("FLOAT", {"default": 360.0, "min": -720.0, "max": 720.0, "step": 1.0}),
                "环绕半径_orbit_radius": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 50.0, "step": 0.1}),
                "相机高度_orbit_height": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "起始角度_start_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
                "推拉速度_dolly_speed": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "摇摄速度_pan_speed": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1}),
                "俯仰速度_tilt_speed": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("CAMERA_POSE",)
    RETURN_NAMES = ("camera_poses_相机轨迹",)
    FUNCTION = "generate"
    CATEGORY = "BSAI/SolarWM-H3 (Deprecated)"
    DESCRIPTION = "[DEPRECATED] Use CameraAttach with orbit_turns + radius"

    def generate(self, 总帧数_num_frames, 相机模式_mode, 环绕角度_orbit_angle,
                 环绕半径_orbit_radius, 相机高度_orbit_height, 起始角度_start_angle,
                 推拉速度_dolly_speed, 摇摄速度_pan_speed, 俯仰速度_tilt_speed):
        # Store params as a dict; ApplyCamera will use them
        params = {
            "num_frames": 总帧数_num_frames,
            "mode": 相机模式_mode,
            "orbit_angle": 环绕角度_orbit_angle,
            "orbit_radius": 环绕半径_orbit_radius,
            "orbit_height": 相机高度_orbit_height,
            "start_angle": 起始角度_start_angle,
            "dolly_speed": 推拉速度_dolly_speed,
            "pan_speed": 摇摄速度_pan_speed,
            "tilt_speed": 俯仰速度_tilt_speed,
        }
        return (params,)


class BSAI_SolarWM_H3_ApplyCamera:
    """[DEPRECATED] Old apply-camera node — use CameraAttach."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_模型": ("MODEL",),
                "camera_poses_相机轨迹": ("CAMERA_POSE",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model_模型",)
    FUNCTION = "apply"
    CATEGORY = "BSAI/SolarWM-H3 (Deprecated)"
    DESCRIPTION = "[DEPRECATED] Use CameraAttach (it combines load + trajectory + apply)"

    def apply(self, model_模型, camera_poses_相机轨迹):
        raise RuntimeError(
            "[BSAI-SolarWM-H3] BSAI_SolarWM_H3_ApplyCamera is deprecated.\n"
            "The new v2.0 uses a single BSAI_SolarWM_H3_CameraAttach node\n"
            "that combines model + orbit params + latent + positive in one step.\n"
            "See workflows/SolarWM_H3_Basic.json for the updated workflow."
        )


# Register deprecated nodes
NODE_CLASS_MAPPINGS.update({
    "BSAI_SolarWM_H3_Loader": BSAI_SolarWM_H3_Loader,
    "BSAI_SolarWM_H3_CameraTrajectory": BSAI_SolarWM_H3_CameraTrajectory,
    "BSAI_SolarWM_H3_ApplyCamera": BSAI_SolarWM_H3_ApplyCamera,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    "BSAI_SolarWM_H3_Loader": "BSAI SolarWM-H3 Loader [DEPRECATED]",
    "BSAI_SolarWM_H3_CameraTrajectory": "BSAI SolarWM-H3 Camera Trajectory [DEPRECATED]",
    "BSAI_SolarWM_H3_ApplyCamera": "BSAI SolarWM-H3 Apply Camera [DEPRECATED]",
})
