// ComPyter — Jupyter Notebook node web extension.
//
// One custom DOM widget owns the whole node body: status bar, code editor,
// buttons, a draggable divider, and the output area. ComfyUI's auto-created
// `code` widget is hidden but kept around so its `.value` saves with the
// workflow; we sync our textarea to it.

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
    name: "ComPyter.JupyterNotebook",
    async nodeCreated(node) {
        if (node.comfyClass !== "JupyterNotebook") return;
        attachNotebookUI(node);
    },
});

// Dynamic IO: matches Python's JupyterNotebook.SLOTS. Slot 0 is required
// and always visible; slots 1..N appear as the trailing slot gets wired
// (rgthree-style), and the trailing empty ones get reclaimed when their
// predecessor disconnects.
const SLOT_NAMES = ["value", "b", "c", "d", "e", "f", "g", "h", "i", "j"];
const MAX_SLOTS = SLOT_NAMES.length;

function lastWiredSlot(node) {
    let last = -1;
    (node.inputs || []).forEach((inp, i) => {
        if (inp && inp.link != null) last = Math.max(last, i);
    });
    (node.outputs || []).forEach((out, i) => {
        if (out && out.links && out.links.length > 0) last = Math.max(last, i);
    });
    return last;
}

function syncDynamicSlots(node) {
    if (!node.inputs || !node.outputs) return;
    const desired = Math.min(
        Math.max(lastWiredSlot(node) + 2, 1),
        MAX_SLOTS,
    );

    // Inputs: grow to `desired`, shrink trailing unwired ones (never slot 0).
    while (node.inputs.length < desired && node.inputs.length < MAX_SLOTS) {
        node.addInput(SLOT_NAMES[node.inputs.length], "*");
    }
    while (node.inputs.length > desired) {
        const i = node.inputs.length - 1;
        if (i <= 0) break;
        if (node.inputs[i] && node.inputs[i].link != null) break;
        node.removeInput(i);
    }

    // Outputs: mirror.
    while (node.outputs.length < desired && node.outputs.length < MAX_SLOTS) {
        node.addOutput(SLOT_NAMES[node.outputs.length], "*");
    }
    while (node.outputs.length > desired) {
        const i = node.outputs.length - 1;
        if (i <= 0) break;
        const links = node.outputs[i] && node.outputs[i].links;
        if (links && links.length > 0) break;
        node.removeOutput(i);
    }

    node.setDirtyCanvas?.(true, true);
}

function mkBtn(label) {
    const b = document.createElement("button");
    b.textContent = label;
    b.style.cssText = `
        background: #2a2a2a; color: #ddd; border: 1px solid #444;
        border-radius: 2px; padding: 4px 10px; cursor: pointer;
        font-family: inherit; font-size: 11px;
    `;
    b.onmouseenter = () => (b.style.background = "#383838");
    b.onmouseleave = () => (b.style.background = "#2a2a2a");
    return b;
}

function stripAnsi(s) {
    return (s || "").replace(/\x1b\[[0-9;]*m/g, "");
}

// Matches Python's _sanitize_label: NFKD normalize -> drop non-ASCII ->
// whitespace to underscore -> drop remaining punctuation.
function sanitizeLabel(s, fallback) {
    fallback = fallback || "notebook";
    if (!s) return fallback;
    let out = String(s).normalize("NFKD")
        .replace(/[^\x00-\x7F]/g, "")
        .trim()
        .replace(/\s+/g, "_")
        .replace(/[^\w-]/g, "");
    return out || fallback;
}

// CodeMirror 6 is loaded lazily from esm.sh, following the official docs
// pattern: no version pins, no ?bundle, no ?deps. esm.sh resolves the dep
// graph itself so all three packages share a single @codemirror/state and
// @codemirror/view module -- otherwise CM's instanceof checks blow up with
// "Unrecognized extension value".
let _cmPromise = null;
function loadCodeMirror() {
    if (_cmPromise) return _cmPromise;
    _cmPromise = Promise.all([
        import("https://esm.sh/codemirror"),
        import("https://esm.sh/@codemirror/lang-python"),
        import("https://esm.sh/@codemirror/theme-one-dark"),
    ]).then(([cm, lang, theme]) => ({
        EditorView: cm.EditorView,
        basicSetup: cm.basicSetup,
        python: lang.python,
        oneDark: theme.oneDark,
    }));
    return _cmPromise;
}

/**
 * Editor with Python syntax highlighting via CodeMirror, with graceful
 * fallback to a plain textarea while CodeMirror loads (or if it fails).
 * Returns { element, get value, set value, focus }.
 */
function makeEditor(initialCode, onShiftEnter) {
    const wrap = document.createElement("div");
    wrap.style.cssText = `
        flex: 1 1 60px; min-height: 40px;
        display: flex; flex-direction: column;
        background: #1b1b1b;
        border: 1px solid #333; border-radius: 2px;
        overflow: hidden; box-sizing: border-box;
        font-family: ui-monospace, Consolas, monospace; font-size: 12px;
    `;

    // Fallback textarea -- visible immediately, replaced by CodeMirror if it loads.
    const ta = document.createElement("textarea");
    ta.placeholder = "# value, b, c, d, e, label injected. Shift+Enter to run.";
    ta.spellcheck = false;
    ta.value = initialCode || "";
    ta.style.cssText = `
        flex: 1 1 auto; width: 100%;
        background: transparent; color: #eee;
        border: none; outline: none;
        padding: 6px;
        font: inherit;
        resize: none; box-sizing: border-box;
    `;
    ta.addEventListener("keydown", (e) => {
        if (e.shiftKey && e.key === "Enter") {
            e.preventDefault();
            onShiftEnter();
        }
    });
    wrap.appendChild(ta);

    let cm = null;

    const api = {
        element: wrap,
        get value() {
            return cm ? cm.state.doc.toString() : ta.value;
        },
        set value(v) {
            if (typeof v !== "string") return;
            if (cm) {
                const cur = cm.state.doc.toString();
                if (cur !== v) {
                    cm.dispatch({ changes: { from: 0, to: cur.length, insert: v } });
                }
            } else if (ta.value !== v) {
                ta.value = v;
            }
        },
        focus() {
            if (cm) cm.focus();
            else ta.focus();
        },
    };

    loadCodeMirror().then((CM) => {
        const startDoc = ta.value;  // capture in case user typed during load
        cm = new CM.EditorView({
            doc: startDoc,
            extensions: [
                CM.basicSetup,
                CM.python(),
                CM.oneDark,
                CM.EditorView.theme({
                    "&": { height: "100%", fontSize: "12px" },
                    ".cm-scroller": { fontFamily: "ui-monospace, Consolas, monospace" },
                    ".cm-content": { padding: "6px 0" },
                    ".cm-gutters": { background: "#1b1b1b", borderRight: "1px solid #2a2a2a" },
                }),
            ],
            parent: wrap,
        });
        // Shift+Enter via raw DOM listener -- avoids importing @codemirror/view
        // separately, which would create a duplicate state/view module instance.
        cm.contentDOM.addEventListener("keydown", (e) => {
            if (e.shiftKey && e.key === "Enter") {
                e.preventDefault();
                onShiftEnter();
            }
        });
        ta.remove();
        console.info("Compyter: CodeMirror 6 (lang-python, one-dark) ready.");
    }).catch((e) => {
        console.warn("Compyter: CodeMirror failed to load, using plain textarea.", e);
    });

    return api;
}

function removeWidgetDOM(widget) {
    // Try every plausible reference to the multiline widget's DOM and remove
    // it (or its ComfyUI-positioned wrapper) from the document.
    const candidates = [
        widget.element,
        widget.inputEl,
        widget.options?.element,
    ].filter(Boolean);
    for (const el of candidates) {
        let target = el;
        let parent = target.parentNode;
        // Walk up to find ComfyUI's positioned wrapper (absolute/fixed) and
        // remove that, so any sibling resize-handle / decorations go with it.
        while (parent && parent !== document.body) {
            const pos = parent.style?.position;
            if (pos === "absolute" || pos === "fixed") {
                target = parent;
                break;
            }
            parent = parent.parentNode;
        }
        if (target.parentNode) target.parentNode.removeChild(target);
    }
}

function buildCustomCodeWidget(editor) {
    // Stand-in for the auto-generated multiline STRING widget. ComfyUI sees
    // this in node.widgets and uses its `value` for prompt-API conversion
    // and workflow save/load -- but its computeSize is zero and it has no
    // draw, so the canvas layout allocates no slot for it.
    const w = {
        type: "compyter_code",
        name: "code",
        options: {},
        computeSize: () => [0, -4],
        draw: () => {},
    };
    Object.defineProperty(w, "value", {
        get: () => editor.value,
        set: (v) => {
            if (typeof v === "string" && editor.value !== v) editor.value = v;
        },
        configurable: true,
    });
    return w;
}

function attachNotebookUI(node) {
    // Snapshot the auto-created multiline `code` widget so we can replace it.
    const autoCode = node.widgets?.find((w) => w.name === "code");
    const initialCode = autoCode?.value || "";
    const codeIndex = autoCode ? node.widgets.indexOf(autoCode) : -1;
    if (autoCode) {
        removeWidgetDOM(autoCode);
        if (codeIndex >= 0) node.widgets.splice(codeIndex, 1);
    }

    // Hide the legacy `label` widget — label is now derived from node.title.
    const labelWidget = node.widgets?.find((w) => w.name === "label");
    if (labelWidget) {
        labelWidget.computeSize = () => [0, -4];
        labelWidget.type = "compyter_hidden_label";
    }

    // Dynamic IO: collapse to one trailing empty slot now (rAF so it lands
    // after ComfyUI restores saved connections on workflow load), then keep
    // it in sync whenever a wire is added or removed.
    requestAnimationFrame(() => syncDynamicSlots(node));
    const origConnChange = node.onConnectionsChange;
    node.onConnectionsChange = function (slotType, slot, connected, link_info, ioSlot) {
        if (origConnChange) origConnChange.apply(this, arguments);
        syncDynamicSlots(this);
    };

    // ---- root: single column flex container that owns the whole node body
    const root = document.createElement("div");
    root.style.cssText = `
        display: flex; flex-direction: column; width: 100%; height: 100%;
        padding: 4px; gap: 4px;
        font-family: ui-monospace, Consolas, monospace; font-size: 12px;
        color: #ddd; box-sizing: border-box; min-height: 0;
    `;

    // ---- status bar (fixed)
    const statusBar = document.createElement("div");
    statusBar.style.cssText = `
        flex: 0 0 auto;
        display: flex; align-items: center; gap: 8px;
        padding: 2px 4px; background: #1f1f1f; border-radius: 2px;
        font-size: 11px;
    `;
    const statusDot = document.createElement("span");
    statusDot.style.cssText =
        "width: 8px; height: 8px; border-radius: 50%; background: #666;";
    const statusText = document.createElement("span");
    statusText.textContent = "kernel: unknown";
    statusBar.appendChild(statusDot);
    statusBar.appendChild(statusText);
    root.appendChild(statusBar);

    // ---- code editor (CodeMirror with Python highlighting; flex item)
    const editor = makeEditor(initialCode, () => execute());
    root.appendChild(editor.element);

    // ---- buttons row (fixed)
    const buttonRow = document.createElement("div");
    buttonRow.style.cssText = "flex: 0 0 auto; display: flex; gap: 4px;";
    const runBtn = mkBtn("Run (⇧⏎)");
    const resumeBtn = mkBtn("Resume ▶");
    resumeBtn.style.background = "#2a5a2a";
    resumeBtn.style.borderColor = "#3a7a3a";
    const clearBtn = mkBtn("Clear");
    buttonRow.appendChild(runBtn);
    buttonRow.appendChild(resumeBtn);
    buttonRow.appendChild(clearBtn);
    root.appendChild(buttonRow);

    // ---- divider (draggable)
    const divider = document.createElement("div");
    divider.title = "Drag to resize editor / output";
    divider.style.cssText = `
        flex: 0 0 6px;
        background: #2a2a2a;
        cursor: row-resize;
        border-radius: 2px;
        margin: 2px 0;
        transition: background 0.1s;
    `;
    divider.onmouseenter = () => (divider.style.background = "#4a4a4a");
    divider.onmouseleave = () => (divider.style.background = "#2a2a2a");
    root.appendChild(divider);

    // ---- output (grows; counterpart of editor in the divider ratio)
    const output = document.createElement("div");
    output.style.cssText = `
        flex: 1 1 60px; min-height: 40px;
        background: #111; border: 1px solid #2a2a2a; border-radius: 2px;
        padding: 4px;
        overflow: auto; white-space: pre-wrap; word-wrap: break-word;
    `;
    root.appendChild(output);

    // ----------- divider drag: adjust editor flex-grow against output -----
    let dragging = false;
    let startY = 0;
    let startEditorH = 0;
    let startOutputH = 0;
    divider.addEventListener("mousedown", (e) => {
        dragging = true;
        startY = e.clientY;
        startEditorH = editor.element.offsetHeight;
        startOutputH = output.offsetHeight;
        document.body.style.cursor = "row-resize";
        e.preventDefault();
    });
    const onMove = (e) => {
        if (!dragging) return;
        const delta = e.clientY - startY;
        const newEditorH = Math.max(40, startEditorH + delta);
        const newOutputH = Math.max(40, startOutputH - delta);
        // Express as flex-grow ratio so the split survives node resizes.
        const ratio = newEditorH / newOutputH;
        editor.element.style.flex = `${ratio} 1 40px`;
        output.style.flex = `1 1 40px`;
    };
    const onUp = () => {
        if (dragging) {
            dragging = false;
            document.body.style.cursor = "";
        }
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);

    // ----------- code widget: insert the editor-backed replacement ----------
    // value reads/writes go straight to editor.value, so workflow save/load
    // and prompt-API conversion all just work. computeSize=[0,-4] means
    // ComfyUI's canvas layout allocates no slot for it.
    if (codeIndex >= 0) {
        const customCode = buildCustomCodeWidget(editor);
        const safeIndex = Math.min(codeIndex, node.widgets.length);
        node.widgets.splice(safeIndex, 0, customCode);
    }

    // ----------- output rendering ------------------------------------------
    function renderOutputs(outs) {
        output.innerHTML = "";
        for (const o of outs || []) {
            const block = document.createElement("div");
            block.style.cssText = "margin: 2px 0;";
            if (o.type === "stream") {
                block.style.color = o.name === "stderr" ? "#ff8a8a" : "#dcdcdc";
                block.textContent = o.text;
            } else if (o.type === "error") {
                block.style.color = "#ff6464";
                const tb = (o.traceback || []).join("\n");
                block.textContent = stripAnsi(tb || `${o.ename}: ${o.evalue}`);
            } else if (o.type === "execute_result" || o.type === "display_data") {
                const data = o.data || {};
                const imgMime = ["image/png", "image/jpeg", "image/gif"]
                    .find((m) => data[m]);
                if (imgMime) {
                    const img = document.createElement("img");
                    img.src = `data:${imgMime};base64,${data[imgMime]}`;
                    img.style.maxWidth = "100%";
                    img.style.borderRadius = "2px";
                    block.appendChild(img);
                } else if (data["image/svg+xml"]) {
                    const wrap = document.createElement("div");
                    wrap.style.maxWidth = "100%";
                    wrap.innerHTML = data["image/svg+xml"];
                    block.appendChild(wrap);
                } else if (data["text/html"]) {
                    const wrap = document.createElement("div");
                    wrap.innerHTML = data["text/html"];
                    block.appendChild(wrap);
                } else if (data["text/plain"]) {
                    block.style.color = "#a3e3a3";
                    block.textContent = data["text/plain"];
                }
            }
            output.appendChild(block);
        }
        output.scrollTop = output.scrollHeight;
    }

    // ----------- execute / resume / clear ----------------------------------
    async function execute() {
        const code = editor.value;
        if (!code.trim()) return;
        const sessionWidget = node.widgets?.find((w) => w.name === "session");
        const session = (sessionWidget?.value || "default").toString();
        const label = sanitizeLabel(node.title);
        runBtn.disabled = true;
        const orig = runBtn.textContent;
        runBtn.textContent = "Running...";
        try {
            const res = await api.fetchApi("/compyter/execute", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ code, session, label }),
            });
            const json = await res.json();
            if (json.ok === false) {
                renderOutputs([{
                    type: "error",
                    ename: "BackendError",
                    evalue: json.error || "unknown",
                    traceback: [],
                }]);
            } else {
                renderOutputs(json.outputs);
            }
        } catch (e) {
            renderOutputs([{
                type: "error",
                ename: "NetworkError",
                evalue: String(e),
                traceback: [],
            }]);
        } finally {
            runBtn.disabled = false;
            runBtn.textContent = orig;
        }
    }

    async function resume() {
        try {
            await api.fetchApi("/compyter/resume", { method: "POST" });
        } catch (e) {
            console.error("Compyter resume failed", e);
        }
    }

    runBtn.onclick = execute;
    resumeBtn.onclick = resume;
    clearBtn.onclick = () => (output.innerHTML = "");
    // Shift+Enter is wired inside makeEditor (forwarded to execute via callback).

    // ----------- status / outputs polling ----------------------------------
    async function pollStatus() {
        try {
            const res = await api.fetchApi("/compyter/status");
            const s = await res.json();
            if (!s.started) {
                statusDot.style.background = "#666";
                statusText.textContent = "kernel: not started";
                resumeBtn.disabled = true;
                resumeBtn.style.opacity = 0.4;
            } else if (s.paused) {
                statusDot.style.background = "#ffae00";
                statusText.textContent = `paused${s.label ? " @ " + s.label : ""}`;
                resumeBtn.disabled = false;
                resumeBtn.style.opacity = 1.0;
            } else {
                statusDot.style.background = "#4caf50";
                statusText.textContent = "kernel: ready";
                resumeBtn.disabled = true;
                resumeBtn.style.opacity = 0.4;
            }
        } catch (e) {
            statusDot.style.background = "#666";
            statusText.textContent = "kernel: ?";
        }
    }

    async function pollOutputs() {
        try {
            const id = encodeURIComponent(String(node.id));
            const res = await api.fetchApi("/compyter/outputs?node_id=" + id);
            const json = await res.json();
            if (json.outputs && json.outputs.length) {
                renderOutputs(json.outputs);
            }
        } catch (e) {
            // ignore
        }
    }

    pollStatus();
    pollOutputs();
    const pollId = setInterval(() => {
        pollStatus();
        pollOutputs();
    }, 1000);

    // ----------- cleanup on removal ----------------------------------------
    const origOnRemoved = node.onRemoved;
    node.onRemoved = function () {
        clearInterval(pollId);
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        if (origOnRemoved) origOnRemoved.apply(this, arguments);
    };

    node.addDOMWidget("notebook_panel", "div", root, { serialize: false });

    if (!node.size || node.size[0] < 420) {
        node.size = [420, 580];
    }
}
