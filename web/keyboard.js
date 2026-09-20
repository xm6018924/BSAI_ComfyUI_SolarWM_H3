import { app } from "../../scripts/app.js";

const NODE_TYPES = [
    "BSAI_SolarWM_H3_Generate",
    "BSAI_SolarWM_H3_CameraAttach",
];

const ext = {
    name: "BSAI.SolarWM-H3.Keyboard",
    async setup() {
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
                    if (w.inputEl === active) {
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
                        if (w.callback) w.callback(v);
                        node.setDirtyCanvas?.(true, true);
                        e.preventDefault();
                        e.stopPropagation();
                        return;
                    }
                }
            }
        });
    },
};

app.registerExtension(ext);