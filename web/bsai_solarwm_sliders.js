/**
 * BSAI SolarWM-H3 camera trajectory centered-slider.
 *
 * For the "BSAI_SolarWM_H3_Generate" (一键生成) node this extension
 * installs a TRUE ZERO-CENTERED horizontal slider DOM-widget on the
 * right side of every camera trajectory FLOAT widget.
 *
 * Strategy that works in ComfyUI v40 (Vue/canvas-mixed frontend):
 */
// Top-level sentinel — if this file is loaded by the frontend at all,
// this line shows up in the browser console immediately.
console.log("[BSAI-SolarWM] extension script loaded");

/**
 * For the "BSAI_SolarWM_H3_Generate" (一键生成) node this extension
 * installs a TRUE ZERO-CENTERED horizontal slider DOM-widget on the
 * right side of every camera trajectory FLOAT widget.
 *
 * Strategy that works in ComfyUI v40 (Vue/canvas-mixed frontend):
 *
 *   - Use `node.addDOMWidget()` — an official ComfyUI / LiteGraph API
 *     that adds a real HTML <div> as a widget. v40's Vue-based widget
 *     layer treats it like any other widget (positions it via
 *     `widget.y` / `widget.computedHeight`, hides it on collapse,
 *     removes it on node removal, etc.). Crucially this DOES NOT touch
 *     the underlying FLOAT widget at all, so the original inline
 *     number editor and +/- arrow buttons keep working exactly as
 *     before.
 *   - Slider drag updates `widget.value` and calls `widget.callback`
 *     so the canvas / graph state stays in sync.
 *   - Keyboard ↑↓←→ step the most-recently-clicked slider by
 *     `widget.options.step`.
 *
 * If `node.addDOMWidget` is unavailable (very old frontend), the
 * extension becomes a no-op.
 */
import { app } from "../../../scripts/app.js";

const TARGET_NODE = "BSAI_SolarWM_H3_Generate";

const SLIDER_FIELDS = [
    "orbit_turns",
    "radius",
    "radius_end",
    "height",
    "height_end",
    "start_angle",
    "pan_speed",
    "tilt_speed",
    "look_at_y",
];

// ---------------------------------------------------------------------------
// Math
// ---------------------------------------------------------------------------

function clamp(v, lo, hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

function roundToStep(v, step, min) {
    if (!step || step <= 0) return v;
    const k = Math.round((v - min) / step);
    const snapped = min + k * step;
    return Math.abs(snapped) < step * 1e-6 ? 0 : snapped;
}

// ---------------------------------------------------------------------------
// Slider DOM element
// ---------------------------------------------------------------------------

function makeSliderEl(name) {
    const root = document.createElement("div");
    root.className = "bsai-solarwm-slider";
    root.dataset.bsaiSolarwmName = name;
    Object.assign(root.style, {
        position: "absolute",
        right: "26px",
        top: "50%",
        transform: "translateY(-50%)",
        width: "44%",
        minWidth: "60px",
        height: "14px",
        display: "flex",
        alignItems: "center",
        cursor: "ew-resize",
        zIndex: "5",
        userSelect: "none",
        pointerEvents: "auto",
        touchAction: "none",
    });

    const track = document.createElement("div");
    Object.assign(track.style, {
        position: "relative",
        width: "100%",
        height: "6px",
        background: "rgba(0,0,0,0.55)",
        borderRadius: "2px",
        pointerEvents: "none",
    });
    root.appendChild(track);

    const tick = document.createElement("div");
    Object.assign(tick.style, {
        position: "absolute",
        left: "50%",
        top: "-3px",
        width: "2px",
        height: "12px",
        background: "#ffffff",
        transform: "translateX(-50%)",
        pointerEvents: "none",
    });
    track.appendChild(tick);

    const fill = document.createElement("div");
    Object.assign(fill.style, {
        position: "absolute",
        top: "0",
        height: "6px",
        pointerEvents: "none",
    });
    track.appendChild(fill);

    const handle = document.createElement("div");
    Object.assign(handle.style, {
        position: "absolute",
        top: "-3px",
        width: "4px",
        height: "12px",
        background: "#ffffff",
        transform: "translateX(-50%)",
        pointerEvents: "none",
        borderRadius: "1px",
    });
    track.appendChild(handle);

    return { root, track, fill, handle };
}

// ---------------------------------------------------------------------------
// Hook every camera parameter on the node
// ---------------------------------------------------------------------------

function setupNode(node) {
    if (node.__bsaiSolarwmSetup) return;
    node.__bsaiSolarwmSetup = true;
    node.__bsaiSolarwmFocused = null;

    if (typeof node.addDOMWidget !== "function") {
        console.log("[BSAI-SolarWM] addDOMWidget not available; extension disabled");
        return;
    }

    // We install sliders lazily on every node refresh; once for each
    // camera parameter. Because the underlying FLOAT widget DOM is
    // created by the Vue layer AFTER our setup, we use raf retry.
    const slInstallAttempts = new Map(); // widgetName -> attempts

    const tryInstall = (widget) => {
        if (!SLIDER_FIELDS.includes(widget.name)) return;
        if (widget.__bsaiSolarwmSliderInstalled) return;
        // We need to wait until the FLOAT widget has a DOM element
        // to attach our absolute-positioned slider to.
        // The widget's underlying DOM is exposed at widget.element in
        // v40 (set by Vue layer). If absent yet, retry later.
        const el = widget.element;
        if (!el) {
            const n = (slInstallAttempts.get(widget.name) || 0) + 1;
            slInstallAttempts.set(widget.name, n);
            if (n < 60) requestAnimationFrame(() => tryInstall(widget));
            return;
        }
        // Ensure relative positioning so the absolute child sits inside
        const cs = getComputedStyle(el);
        if (cs.position === "static") el.style.position = "relative";

        const { root, track, fill, handle } = makeSliderEl(widget.name);

        // ---- visual update ----
        function updateVisual() {
            const origMin = widget.options?.min ?? 0;
            const origMax = widget.options?.max ?? 1;
            const halfRange = Math.max(Math.abs(origMin), Math.abs(origMax));
            const w = track.clientWidth || 1;
            const t = halfRange > 0
                ? clamp(widget.value / halfRange, -1, 1)
                : 0;
            const offsetPx = (w * 0.5) * (1 + t);
            if (widget.value >= 0) {
                fill.style.left = "50%";
                fill.style.width = (offsetPx - w * 0.5) + "px";
                fill.style.background = "#6ea8fe";
            } else {
                fill.style.left = offsetPx + "px";
                fill.style.width = (w * 0.5 - offsetPx) + "px";
                fill.style.background = "#ff8a8a";
            }
            handle.style.left = offsetPx + "px";
        }

        // ---- pointer interaction ----
        function pointerToValue(clientX) {
            const r = track.getBoundingClientRect();
            const t = clamp((clientX - r.left) / r.width, 0, 1);
            const origMin = widget.options?.min ?? 0;
            const origMax = widget.options?.max ?? 1;
            const halfRange = Math.max(Math.abs(origMin), Math.abs(origMax));
            const step = widget.options?.step || 0.01;
            let v = (t * 2 - 1) * halfRange;
            v = clamp(roundToStep(v, step, origMin), origMin, origMax);
            return v;
        }

        function commitValue(v) {
            if (v === widget.value) return;
            widget.value = v;
            widget.callback?.(widget.value);
            // Sync the inline <input> the Vue layer renders for this
            // widget so it shows the new value.
            const input = el.querySelector("input,textarea");
            if (input) {
                input.value = v;
                input.dispatchEvent(new Event("input", { bubbles: true }));
                input.dispatchEvent(new Event("change", { bubbles: true }));
            }
        }

        let dragging = false;
        const onDown = (e) => {
            dragging = true;
            try { root.setPointerCapture(e.pointerId); } catch (_) {}
            node.__bsaiSolarwmFocused = widget.name;
            commitValue(pointerToValue(e.clientX));
            updateVisual();
            e.preventDefault();
            e.stopPropagation();
        };
        const onMove = (e) => {
            if (!dragging) return;
            commitValue(pointerToValue(e.clientX));
            updateVisual();
            e.preventDefault();
            e.stopPropagation();
        };
        const onUp = (e) => {
            if (!dragging) return;
            dragging = false;
            try { root.releasePointerCapture(e.pointerId); } catch (_) {}
            e.preventDefault();
            e.stopPropagation();
        };

        root.addEventListener("pointerdown", onDown);
        root.addEventListener("pointermove", onMove);
        root.addEventListener("pointerup", onUp);
        root.addEventListener("pointercancel", onUp);

        el.appendChild(root);
        widget.__bsaiSolarwmSliderInstalled = true;
        widget.__bsaiSolarwmSliderEl = root;
        widget.__bsaiSolarwmUpdateVisual = updateVisual;

        // Hook into the widget's callback so we repaint when the
        // value changes through other UI (inline editor, +/- arrows).
        const origCb = widget.callback;
        widget.callback = function (...args) {
            const r = origCb?.apply(this, args);
            updateVisual();
            return r;
        };

        // Keep visual synced with widget size changes
        const ro = new ResizeObserver(() => updateVisual());
        ro.observe(el);
        widget.__bsaiSolarwmRO = ro;

        updateVisual();
        console.log("[BSAI-SolarWM] installed slider for", widget.name);
    };

    // Trigger install for every camera parameter widget
    const onWidgetsReady = () => {
        for (const w of node.widgets || []) tryInstall(w);
    };

    // First pass (widgets may already exist)
    onWidgetsReady();

    // Vue layer may add widgets slightly later
    requestAnimationFrame(onWidgetsReady);
    setTimeout(onWidgetsReady, 100);
    setTimeout(onWidgetsReady, 500);

    // When new widgets appear, retry
    if (typeof MutationObserver !== "undefined" && app.canvas?.canvas) {
        let scheduled = false;
        const obs = new MutationObserver(() => {
            if (scheduled) return;
            scheduled = true;
            requestAnimationFrame(() => {
                scheduled = false;
                onWidgetsReady();
            });
        });
        obs.observe(app.canvas.canvas, { childList: true, subtree: true });
        node.__bsaiSolarwmObserver = obs;
    }

    // Keyboard arrow stepping
    if (!node.__bsaiSolarwmKbdInstalled) {
        node.__bsaiSolarwmKbdInstalled = true;
        const handler = (e) => {
            const canvas = app.canvas;
            if (!canvas) return;
            const selected = canvas.selected_nodes;
            const isSelected =
                (selected && Object.values(selected).includes(node))
                || canvas.current_node === node;
            if (!isSelected) return;
            let dir = 0;
            switch (e.key) {
                case "ArrowLeft": case "ArrowDown": dir = -1; break;
                case "ArrowRight": case "ArrowUp": dir = +1; break;
                default: return;
            }
            const tag = (e.target?.tagName || "").toUpperCase();
            if (tag === "INPUT" || tag === "TEXTAREA") return;
            let widget = null;
            if (node.__bsaiSolarwmFocused) {
                widget = (node.widgets || []).find(
                    (w) => w.name === node.__bsaiSolarwmFocused
                        && SLIDER_FIELDS.includes(w.name),
                );
            }
            if (!widget) {
                widget = (node.widgets || []).find(
                    (w) => SLIDER_FIELDS.includes(w.name),
                );
            }
            if (!widget) return;
            const step = widget.options?.step || 0.01;
            const origMin = widget.options?.min ?? 0;
            const origMax = widget.options?.max ?? 1;
            const next = clamp(
                roundToStep(widget.value + dir * step, step, origMin),
                origMin, origMax,
            );
            if (next !== widget.value) {
                widget.value = next;
                widget.callback?.(widget.value);
                node.__bsaiSolarwmFocused = widget.name;
                widget.__bsaiSolarwmUpdateVisual?.();
            }
            e.preventDefault();
            e.stopPropagation();
        };
        window.addEventListener("keydown", handler, true);
        node.__bsaiSolarwmKbdHandler = handler;
    }

    // Cleanup on node removal
    const origOnRemoved = node.onRemoved;
    node.onRemoved = function () {
        for (const w of this.widgets || []) {
            if (w.__bsaiSolarwmSliderEl) w.__bsaiSolarwmSliderEl.remove();
            if (w.__bsaiSolarwmRO) w.__bsaiSolarwmRO.disconnect();
            w.__bsaiSolarwmSliderInstalled = false;
        }
        if (this.__bsaiSolarwmObserver) {
            this.__bsaiSolarwmObserver.disconnect();
            this.__bsaiSolarwmObserver = null;
        }
        if (this.__bsaiSolarwmKbdHandler) {
            window.removeEventListener(
                "keydown",
                this.__bsaiSolarwmKbdHandler,
                true,
            );
        }
        return origOnRemoved?.apply(this, arguments);
    };
}

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "BSAI.SolarWM_H3.CenteredSliders",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name !== TARGET_NODE) return;

        const wrap = (key) => {
            const orig = nodeType.prototype[key];
            nodeType.prototype[key] = function () {
                const r = orig?.apply(this, arguments);
                setupNode(this);
                return r;
            };
        };
        wrap("onNodeCreated");
        wrap("onConfigure");
    },

    async loadedGraphNode(node) {
        if (node?.comfyClass !== TARGET_NODE) return;
        setupNode(node);
    },
});