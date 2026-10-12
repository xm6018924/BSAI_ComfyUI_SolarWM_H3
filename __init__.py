# === BSAI H3 VAE 全量加载统一补丁（宿主 BSAI-ComfyUI-Sol-H3）===
try:
    import sys as _sys, os as _os
    _SOL = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                         "..", "BSAI-ComfyUI-Sol-H3")
    if _SOL not in _sys.path:
        _sys.path.insert(0, _SOL)
    import bsai_h3_vae_full as _h3v
    _h3v.apply_global_patch("BSAI_ComfyUI_SolarWM_H3")
except Exception:
    pass

from .bsai_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
import logging, os

log = logging.getLogger("BSAI-SolarWM")

# Force-load web directory registration (ComfyUI only does it
# automatically if WEB_DIRECTORY is defined at import time).
_here = os.path.dirname(os.path.abspath(__file__))
_web_dir = os.path.join(_here, "web")
if os.path.isdir(_web_dir):
    log.info("[BSAI-SolarWM-H3] web directory ready: %s", _web_dir)
    for fn in sorted(os.listdir(_web_dir)):
        if fn.endswith(".js"):
            log.info("[BSAI-SolarWM-H3]   - web asset: %s", fn)
else:
    log.warning("[BSAI-SolarWM-H3] web directory missing: %s", _web_dir)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
__version__ = "2.2.1"

# Auto-mounted by ComfyUI as a static frontend extension directory.
# Contains bsai_solarwm_sliders.js which adds 0-centered horizontal
# sliders + Arrow-key stepping to the camera-trajectory FLOAT widgets of
# the "BSAI_SolarWM_H3_Generate" node.
WEB_DIRECTORY = "./web"