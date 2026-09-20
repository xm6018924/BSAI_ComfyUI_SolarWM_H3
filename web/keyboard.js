import { app } from "../../scripts/app.js";

const NODE_TYPES = [
    "BSAI_SolarWM_H3_Generate",
    "BSAI_SolarWM_H3_CameraAttach",
];

function addSliderToWidget(widget, node) {
    if (!widget || widget._bsaiSlider) return;
    if (widget.type !== "number" && widget.type !== undefined) {
        // Only target number widgets
    }
    if (widget.type && widget.type !== "number" && widget.type !== "slider" && widget.type !== undefined) {
        return;
    }
    // Skip combo widgets
    if (widget.type === "combo" || widget.options?.values) return;

    const min = widget.options?.min;
    const max = widget.options?.max;
    if (min === undefined || max === undefined) return;

    const step = widget.options?.step || 0.01;

    // Find widget DOM element
    const el = widget.inputEl || widget.element;
    if (!el) return;
    const container = el.parentElement;
    if (!container) return;

    // Create slider
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = min;
    slider.max = max;
    slider.step = step;
    slider.value = widget.value ?? 0;
    slider.className = "bsai-cam-slider";
    slider.style.cssText = "width:100%;height:8px;margin:2px 0 4px 0;cursor:pointer;-webkit-appearance:none;background:linear-gradient(to right,#3b82f6,#ef4444);border-radius:4px;outline:none;display:block;";

    slider.addEventListener("pointerdown", (e) => e.stopPropagation());
    slider.addEventListener("wheel", (e) => e.stopPropagation(), { passive: false });

    slider.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        widget.value = val;
        if (widget.inputEl) widget.inputEl.value = val;
        if (widget.callback) widget.callback(val);
        node.setDirtyCanvas?.(true, true);
    });

    // Sync input -> slider
    const syncInput = () => { slider.value = widget.value ?? 0; };
    el.addEventListener("change", syncInput);
    el.addEventListener("input", syncInput);

    container.appendChild(slider);
    widget._bsaiSlider = slider;
}

function setupNodeSliders(node) {
    if (!node?.widgets) return;
    for (const w of node.widgets) {
        addSliderToWidget(w, node);
    }
}

const ext = {
    name: "BSAI.SolarWM-H3.Slider",
    async setup() {
        // Keyboard arrow keys
        document.addEventListener("keydown", (e) => {
            const tag = e.target.tagName;
            if (tag === "INPUT" || tag === "TEXTAREA") return;
            if (e.ctrlKey || e.metaKey || e.altKey) return;

            const selected = app.graph?._nodes?.filter(n => n.selected);
            if (!selected?.length) return;

            for (const node of selected) {
                if (!NODE_TYPES.includes(node.type)) continue;
                for (const w of node.widgets || []) {
                    if (w.type !== "number" || w.options?.values) continue;
                    const active = document.activeElement;
                    if (w.inputEl === active || (w._bsaiSlider && w._bsaiSlider === active)) {
                        const step = w.options?.step || 0.01;
                        const fine = step * 0.1;
                        const min = w.options?.min ?? -Infinity;
                        const max = w.options?.max ?? Infinity;
                        let v = w.value ?? 0;
                        if (e.key === "ArrowUp") v = Math.min(max, v + step);
                        else if (e.key === "ArrowDown") v = Math.max(min, v - step);
                        else if (e.key === "ArrowLeft") v = Math.max(min, v - fine);
                        else if (e.key === "ArrowRight") v = Math.min(max, v + fine);
                        else continue;
                        w.value = v;
                        if (w.inputEl) w.inputEl.value = v;
                        if (w._bsaiSlider) w._bsaiSlider.value = v;
                        if (w.callback) w.callback(v);
                        node.setDirtyCanvas?.(true, true);
                        e.preventDefault();
                        e.stopPropagation();
                        return;
                    }
                }
            }
        });

        // Periodically check for our nodes (handles workflow load)
        setInterval(() => {
            const nodes = app.graph?._nodes || [];
            for (const n of nodes) {
                if (NODE_TYPES.includes(n.type)) {
                    setupNodeSliders(n);
                }
            }
        }, 1000);
    },

    nodeCreated(node) {
        if (!NODE_TYPES.includes(node.type)) return;
        setTimeout(() => setupNodeSliders(node), 200);
        setTimeout(() => setupNodeSliders(node), 500);
        setTimeout(() => setupNodeSliders(node), 1000);
    },
};

app.registerExtension(ext);