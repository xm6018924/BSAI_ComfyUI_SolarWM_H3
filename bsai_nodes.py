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

class BSAI_SolarWM_H3_Loader:
    """Combined loader: UNETLoader + turbo LoRA + SolarWM LoRA in one node."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        unet_models = folder_paths.get_filename_list("diffusion_models")
        lora_models = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "base_model": (sorted(unet_models), {
                    "tooltip": "H3 diffusion model / H3 扩散模型",
                }),
                "turbo_lora": (sorted(lora_models), {
                    "tooltip": "4-step distill LoRA / 4步蒸馏 LoRA",
                }),
                "turbo_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Turbo LoRA strength / 蒸馏 LoRA 强度",
                }),
                "solarwm_lora": (sorted(lora_models), {
                    "tooltip": "SolarWM camera adapter LoRA / SolarWM 相机适配器 LoRA",
                }),
                "solarwm_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "SolarWM LoRA strength / SolarWM LoRA 强度",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "BSAI/SolarWM-H3"

    def load(self, base_model, turbo_lora, turbo_strength,
             solarwm_lora, solarwm_strength):
        import comfy.sd
        import folder_paths

        model_options = {}
        model = comfy.sd.load_diffusion_model(
            folder_paths.get_full_path("diffusion_models", base_model),
            model_options=model_options,
        )

        turbo_path = folder_paths.get_full_path("loras", turbo_lora)
        if turbo_path and turbo_strength > 0:
            model_lora = comfy.utils.load_torch_file(turbo_path, safe_load=True)
            model = model.clone()
            model.add_patches(model_lora, strength_patch=turbo_strength,
                              strength_model=turbo_strength)
            del model_lora

        solarwm_path = folder_paths.get_full_path("loras", solarwm_lora)
        if solarwm_path and solarwm_strength > 0:
            model_lora = comfy.utils.load_torch_file(solarwm_path, safe_load=True)
            model = model.clone()
            model.add_patches(model_lora, strength_patch=solarwm_strength,
                              strength_model=solarwm_strength)
            del model_lora

        return (model,)


NODE_CLASS_MAPPINGS = {
    "BSAI_SolarWM_H3_Loader": BSAI_SolarWM_H3_Loader,
    "BSAI_SolarWM_H3_CameraAttach": BSAI_SolarWM_H3_CameraAttach,
    "BSAI_SolarWM_H3_Generate": BSAI_SolarWM_H3_Generate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_SolarWM_H3_Loader": "BSAI SolarWM-H3 Loader (模型加载)",
    "BSAI_SolarWM_H3_CameraAttach": "BSAI SolarWM-H3 Camera Attach (相机轨迹)",
    "BSAI_SolarWM_H3_Generate": "BSAI SolarWM-H3 Generate (一键生成)",
}
