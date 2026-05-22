"""
ComfyUI custom node: Jupyter Breakpoint.

Inline breakpoint that pauses a workflow and exposes the live input value to
a persistent background Jupyter kernel. Connect a front-end with
`jupyter console --existing <conn_file>`, inspect `value`, call `resume()`
to continue the graph.

One kernel per process, started lazily on first hit, fixed connection file,
shared user_ns mutated in place on each pause -- so a single notebook
attaches once and survives across queue runs.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import re
import signal
import sys
import tempfile
import threading
import time

log = logging.getLogger("comfyui-jupyter-breakpoint")

try:
    import comfy.model_management as _comfy_mm
except Exception:
    _comfy_mm = None


# ---------------------------------------------------------------------------
# Wildcard type: ComfyUI compares input/output type strings with `!=`. By
# making __ne__ always return False, ANY compares equal to every other type
# regardless of which side it sits on (str subclass takes precedence on the
# reflected comparison).
# ---------------------------------------------------------------------------
class AnyType(str):
    def __ne__(self, other):
        return False


ANY = AnyType("*")


# ---------------------------------------------------------------------------
# Singleton kernel state. The pause_event is shared by every breakpoint hit;
# we clear/set it per pause, and the same kernel thread keeps serving.
# ---------------------------------------------------------------------------
_kernel_lock = threading.Lock()
_exec_lock = threading.RLock()   # re-entrant so session ops can wrap execs
_kernel_state: dict = {
    "started": False,
    "user_ns": None,
    "connection_file": None,
    "pause_event": threading.Event(),
    "orig_stdout": None,
    "thread": None,
    "paused": False,
    "paused_label": None,
    "client": None,
    "sessions": {},          # name -> dict of user vars saved out of user_ns
    "active_session": None,  # which session's vars are currently in user_ns
    "ipython_keys": None,    # snapshot of keys IPython owns; user-keys = the rest
}

# Slot + label names are partitioned per session, not shared with IPython.
_SESSION_RESERVED = {"value", "b", "c", "d", "e", "f", "g", "h", "i", "j", "label"}

# IPython-managed convenience names that we never store as session state.
_IPYTHON_NAMES = {"In", "Out", "exit", "quit", "get_ipython", "help", "display"}


def _ensure_ipython_keys():
    """Lazily snapshot the kernel's IPython-owned keys. Anything currently
    present in user_ns (after kernel boot) minus our reserved slot/label
    names counts as IPython-managed and is shared across sessions."""
    if _kernel_state.get("ipython_keys") is not None:
        return
    ns = _kernel_state.get("user_ns") or {}
    _kernel_state["ipython_keys"] = set(ns.keys()) - _SESSION_RESERVED


def _is_user_key(k: str) -> bool:
    """True if key `k` should be partitioned per session."""
    if k.startswith("_"):
        return False
    if k in _IPYTHON_NAMES:
        return False
    ipy_keys = _kernel_state.get("ipython_keys") or set()
    if k in ipy_keys:
        return False
    return True


def _activate_session(name: str) -> None:
    """Make `name` the active session: save current user vars to whichever
    session is currently active, clear them out of user_ns, then load
    `name`'s saved vars in. No-op if `name` is already active."""
    with _exec_lock:
        _ensure_ipython_keys()
        ns = _kernel_state["user_ns"]
        sessions = _kernel_state["sessions"]
        active = _kernel_state.get("active_session")
        if active == name:
            return
        if active is not None:
            sessions[active] = {k: v for k, v in ns.items() if _is_user_key(k)}
        for k in [k for k in list(ns.keys()) if _is_user_key(k)]:
            del ns[k]
        ns.update(sessions.get(name, {}))
        _kernel_state["active_session"] = name


def _save_active_session() -> None:
    """Persist the active session's user-vars from user_ns back to its dict."""
    with _exec_lock:
        active = _kernel_state.get("active_session")
        if active is None:
            return
        ns = _kernel_state["user_ns"]
        _kernel_state["sessions"][active] = {
            k: v for k, v in ns.items() if _is_user_key(k)
        }


def _resolve_connection_file_path() -> str:
    try:
        from jupyter_core.paths import jupyter_runtime_dir
        runtime_dir = jupyter_runtime_dir()
    except Exception:
        runtime_dir = tempfile.gettempdir()
    os.makedirs(runtime_dir, exist_ok=True)
    return os.path.join(runtime_dir, "comfyui_jupyter_breakpoint.json")


def _start_kernel_thread(
    connection_file: str,
    user_ns: dict,
    ready: threading.Event,
    result: dict,
) -> None:
    """Background thread body: spin up an IPython ZMQ kernel that serves forever."""
    try:
        # tornado's IOLoop wraps asyncio; off-main-thread it has no current loop.
        # On Windows, tornado/pyzmq require SelectorEventLoop, not the default
        # ProactorEventLoop. Set per-thread without touching the global policy.
        if sys.platform == "win32":
            asyncio.set_event_loop(asyncio.SelectorEventLoop())
        else:
            asyncio.set_event_loop(asyncio.new_event_loop())

        # IPKernelApp.init_signal installs SIGINT via signal.signal(), which
        # only works on the main thread. Neutralise it across initialize().
        _orig_signal = signal.signal

        def _noop_signal(*args, **kwargs):
            return None

        signal.signal = _noop_signal
        app = None
        try:
            from ipykernel.kernelapp import IPKernelApp

            def _close_failed_app(a):
                """Release ZMQ sockets bound by a partially-initialized app
                so the next retry won't hit 'Address in use' on the same port.
                """
                if a is None:
                    return
                for attr in ("shell_socket", "iopub_socket", "stdin_socket",
                             "control_socket", "heartbeat"):
                    try:
                        sock = getattr(a, attr, None)
                        if sock is None:
                            continue
                        if hasattr(sock, "close"):
                            try:
                                sock.close(linger=0)
                            except TypeError:
                                sock.close()
                    except Exception:
                        pass

            def _wipe_connection_file():
                try:
                    if os.path.exists(connection_file):
                        os.remove(connection_file)
                except Exception:
                    pass

            # IPython startup occasionally iterates dicts (sys.modules,
            # completer state, ...) that other ComfyUI threads mutate
            # concurrently -- "dictionary changed size during iteration".
            # Retry a few times; the racing import settles quickly.
            #
            # Between retries we (a) close sockets the failed app bound and
            # (b) delete the connection file, so the next attempt picks
            # fresh random ports instead of re-binding the prior set.
            last_err: Exception | None = None
            for attempt in range(8):
                _close_failed_app(app)
                app = None
                _wipe_connection_file()
                try:
                    IPKernelApp.clear_instance()
                except Exception:
                    pass
                try:
                    app = IPKernelApp.instance(
                        user_ns=user_ns,
                        connection_file=connection_file,
                    )
                    app.initialize([])
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    msg = str(e)
                    transient = (
                        "dictionary changed size" in msg
                        or "set changed size" in msg
                        or "Address in use" in msg
                        or "Address already in use" in msg
                    )
                    if transient and attempt < 7:
                        time.sleep(0.2 * (attempt + 1))
                        continue
                    raise
            if last_err is not None:
                raise last_err
            try:
                app.write_connection_file()
            except Exception:
                pass
        finally:
            signal.signal = _orig_signal

        result["ok"] = True
        ready.set()
        app.start()  # blocks forever serving the kernel
    except Exception as e:
        result["error"] = repr(e)
        log.exception("Jupyter Breakpoint kernel thread crashed.")
        ready.set()


def _make_resume(pause_event: threading.Event, orig_stdout):
    def resume():
        """Resume the paused ComfyUI graph. Kernel stays alive for the next hit."""
        try:
            orig_stdout.write("[Jupyter Breakpoint] resume() -- continuing graph.\n")
            orig_stdout.flush()
        except Exception:
            pass
        pause_event.set()

    return resume


def _attach_block(connection_file: str, label: str) -> str:
    return (
        f"\n[Jupyter Breakpoint: {label}] paused. Connect a front-end:\n"
        f"  jupyter console   --existing {connection_file}\n"
        f"  jupyter qtconsole --existing {connection_file}     # inline plots\n"
        f"  Lab GUI: EXISTING_CONNECTION_FILE={connection_file} \\\n"
        f"    jupyter lab --KernelProvisionerFactory.default_provisioner_name=existing-provisioner\n"
        f"Then in a cell: inspect `value`, then call resume() to continue the graph.\n"
    )


def _ensure_kernel() -> None:
    """Start the kernel once; subsequent calls are cheap no-ops.

    Raises RuntimeError if the kernel thread failed to bring the kernel up,
    so the node fails fast instead of pausing forever on a dead kernel.
    """
    with _kernel_lock:
        if _kernel_state["started"]:
            return

        # Capture stdout before ipykernel's OutStream redirects it process-wide.
        _kernel_state["orig_stdout"] = sys.stdout

        connection_file = _resolve_connection_file_path()
        # Pre-seed value/label so manual Run before any queue execution
        # doesn't NameError. They get replaced on each queue/pause.
        user_ns: dict = {
            "__name__": "__main__",
            "__doc__": None,
            "value": None,
            "label": "",
        }

        ready = threading.Event()
        result: dict = {}
        thread = threading.Thread(
            target=_start_kernel_thread,
            args=(connection_file, user_ns, ready, result),
            name="comfyui-jupyter-breakpoint",
            daemon=True,
        )
        thread.start()
        ready.wait(timeout=30.0)

        if not result.get("ok"):
            err = result.get("error") or "kernel did not become ready within 30s"
            raise RuntimeError(
                "Jupyter Breakpoint kernel failed to start: "
                f"{err}. Did you `pip install -r requirements.txt` into the "
                "ComfyUI Python env?"
            )

        _kernel_state.update({
            "started": True,
            "user_ns": user_ns,
            "connection_file": connection_file,
            "thread": thread,
        })


# ---------------------------------------------------------------------------
# Kernel client (for the in-node UI to execute code via ZMQ, getting back
# rich Jupyter outputs: stream/execute_result/display_data/error).
# ---------------------------------------------------------------------------
def _get_kernel_client():
    """Lazy ZMQ client connected to our singleton kernel."""
    client = _kernel_state.get("client")
    if client is not None:
        return client
    _ensure_kernel()
    from jupyter_client import BlockingKernelClient
    client = BlockingKernelClient()
    client.load_connection_file(_kernel_state["connection_file"])
    client.start_channels()
    client.wait_for_ready(timeout=15)
    _kernel_state["client"] = client
    return client


def _execute_in_kernel(code: str, timeout: float = 120.0) -> dict:
    """Run code in the kernel; collect IOPub outputs until idle. Serialized."""
    with _exec_lock:
        client = _get_kernel_client()
        # Drain any stale messages from prior runs to avoid mis-attribution.
        while True:
            try:
                client.get_iopub_msg(timeout=0)
            except queue.Empty:
                break
            except Exception:
                break

        msg_id = client.execute(code, store_history=True, allow_stdin=False)
        outputs: list = []
        execution_count = None
        deadline = time.time() + timeout
        while True:
            remaining = max(0.0, deadline - time.time())
            try:
                msg = client.get_iopub_msg(timeout=remaining or 0.1)
            except queue.Empty:
                outputs.append({"type": "error", "ename": "Timeout",
                                "evalue": f"no idle within {timeout}s",
                                "traceback": []})
                break
            except Exception as e:
                outputs.append({"type": "error", "ename": "ClientError",
                                "evalue": repr(e), "traceback": []})
                break

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            mtype = msg.get("msg_type")
            content = msg.get("content", {})
            if mtype == "stream":
                outputs.append({"type": "stream",
                                "name": content.get("name", "stdout"),
                                "text": content.get("text", "")})
            elif mtype == "execute_result":
                execution_count = content.get("execution_count")
                outputs.append({"type": "execute_result",
                                "data": content.get("data", {}),
                                "execution_count": execution_count})
            elif mtype == "display_data":
                outputs.append({"type": "display_data",
                                "data": content.get("data", {})})
            elif mtype == "error":
                outputs.append({"type": "error",
                                "ename": content.get("ename", ""),
                                "evalue": content.get("evalue", ""),
                                "traceback": content.get("traceback", [])})
            elif mtype == "status" and content.get("execution_state") == "idle":
                break

        return {"ok": True, "outputs": outputs, "execution_count": execution_count}


def _set_paused(label: str | None) -> None:
    _kernel_state["paused"] = label is not None
    _kernel_state["paused_label"] = label


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def _do_pause_multi(
    values: dict,
    label: str,
    print_attach_block: bool,
    session: str = "default",
) -> dict:
    """Pause and expose all named slots in `values` to the kernel namespace.

    On resume returns the dict back, reading whatever each name is bound to
    -- so users can mutate any of them in the cell and have the changes
    flow downstream. State is partitioned per `session`.
    """
    _ensure_kernel()
    _activate_session(session)

    pause_event: threading.Event = _kernel_state["pause_event"]
    pause_event.clear()

    ns = _kernel_state["user_ns"]
    for k, v in values.items():
        ns[k] = v
    ns["label"] = label
    ns["resume"] = _make_resume(pause_event, _kernel_state["orig_stdout"])

    if print_attach_block:
        out = _kernel_state["orig_stdout"] or sys.stdout
        try:
            out.write(_attach_block(_kernel_state["connection_file"], label))
            out.flush()
        except Exception:
            print(_attach_block(_kernel_state["connection_file"], label))

    _set_paused(label)
    try:
        # Poll so ComfyUI's "Interrupt" can break the pause; raises
        # InterruptProcessingException out of run() if interrupt was requested.
        while not pause_event.wait(timeout=0.25):
            if _comfy_mm is not None:
                _comfy_mm.throw_exception_if_processing_interrupted()
    finally:
        _set_paused(None)

    out = {k: ns.get(k, v) for k, v in values.items()}
    _save_active_session()
    return out


def _do_pause(value, label: str, print_attach_block: bool, session: str = "default"):
    """Single-slot wrapper around _do_pause_multi for the Breakpoint node."""
    return _do_pause_multi(
        {"value": value}, label, print_attach_block, session=session
    )["value"]


class JupyterBreakpoint:
    """Pause the workflow and expose inputs to a live Jupyter kernel.

    Pro variant: prints a copy-pasteable attach block, expects an external
    front-end (`jupyter console --existing ...`). No in-node UI.

    interactive=True  -> pause + external attach
    interactive=False -> pure passthrough (no code stored on this node)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (ANY, {}),
                "pause": ("BOOLEAN", {"default": True}),
                # `label` is legacy: hidden in the UI, kept so old workflows
                # still load. The real label is the sanitized node title.
                "label": ("STRING", {"default": ""}),
                "session": ("STRING", {"default": "default"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"
    CATEGORY = "debug"
    OUTPUT_NODE = True   # always execute, even with no downstream consumer

    @classmethod
    def IS_CHANGED(cls, value, pause, label, session="default",
                   unique_id=None, extra_pnginfo=None):
        return time.time() if pause else "passthrough"

    def run(self, value, pause, label, session="default",
            unique_id=None, extra_pnginfo=None):
        if not pause:
            return (value,)
        label = _resolve_label_from_workflow(
            extra_pnginfo, unique_id, default="breakpoint"
        )
        return (_do_pause(value, label, print_attach_block=True, session=session),)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# ---------------------------------------------------------------------------
# Label = sanitized node title. NFKD decompose -> drop non-ASCII -> whitespace
# to underscore -> drop remaining punctuation. JS uses an equivalent algorithm
# so server-set `label` (status bar / pause banner) matches what manual Run
# requests send.
# ---------------------------------------------------------------------------
import unicodedata as _unicodedata

_LABEL_WS_RE = re.compile(r"\s+")
_LABEL_PUNCT_RE = re.compile(r"[^\w\-]")


def _sanitize_label(s, default: str = "notebook") -> str:
    if not s:
        return default
    s = _unicodedata.normalize("NFKD", str(s))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = _LABEL_WS_RE.sub("_", s.strip())
    s = _LABEL_PUNCT_RE.sub("", s)
    return s or default


def _resolve_label_from_workflow(extra_pnginfo, unique_id, default: str) -> str:
    """Pull the LiteGraph title for this node id from the embedded workflow
    and sanitize. Falls back to `default` if anything is missing."""
    try:
        if not extra_pnginfo or not unique_id:
            return _sanitize_label(default, default=default)
        nodes = (extra_pnginfo.get("workflow") or {}).get("nodes") or []
        target = str(unique_id)
        for n in nodes:
            if str(n.get("id")) == target:
                return _sanitize_label(n.get("title") or default, default=default)
    except Exception:
        pass
    return _sanitize_label(default, default=default)


def _exec_noninteractive_multi(
    values: dict,
    label: str,
    code: str,
    unique_id=None,
    session: str = "default",
) -> dict:
    """Run `code` against `values`/`label` in the kernel namespace, scoped to
    `session`. Returns a dict with the same keys, reading whatever each is
    bound to after exec. Buffers outputs under `unique_id` for the UI."""
    _ensure_kernel()
    _activate_session(session)
    ns = _kernel_state["user_ns"]
    for k, v in values.items():
        ns[k] = v
    ns["label"] = label
    result = _execute_in_kernel(code)
    outs = result.get("outputs", [])
    if unique_id is not None:
        buf = _kernel_state.setdefault("last_outputs", {})
        buf[str(unique_id)] = outs
    out = {k: ns.get(k, v) for k, v in values.items()}
    _save_active_session()
    for o in outs:
        if o.get("type") == "error":
            tb = _ANSI_RE.sub("", "\n".join(o.get("traceback", [])))
            raise RuntimeError(
                f"JupyterNotebook ({label}) code error:\n"
                f"{tb or (o.get('ename', '') + ': ' + o.get('evalue', ''))}"
            )
    return out


def _exec_noninteractive(value, label: str, code: str, unique_id=None,
                         session: str = "default"):
    """Single-slot wrapper around _exec_noninteractive_multi."""
    return _exec_noninteractive_multi(
        {"value": value}, label, code, unique_id=unique_id, session=session
    )["value"]


def _exec_no_kernel(values: dict, label: str, code: str, unique_id=None) -> dict:
    """Run `code` in a fresh local namespace seeded with `values` + `label`.

    No kernel involvement, no shared state across runs. stdout/stderr are
    captured and buffered for the in-node output panel.
    """
    import contextlib
    import io
    import traceback

    ns = {
        **values,
        "label": label,
        "__name__": "__main__",
        "__builtins__": __builtins__,
    }
    buf_out, buf_err = io.StringIO(), io.StringIO()
    error: Exception | None = None
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        try:
            exec(compile(code, f"<JupyterNotebook:{label}>", "exec"), ns)
        except Exception as e:
            error = e
            traceback.print_exc(file=buf_err)

    outs: list = []
    if buf_out.getvalue():
        outs.append({"type": "stream", "name": "stdout", "text": buf_out.getvalue()})
    if buf_err.getvalue():
        outs.append({"type": "stream", "name": "stderr", "text": buf_err.getvalue()})
    if unique_id is not None:
        _kernel_state.setdefault("last_outputs", {})[str(unique_id)] = outs

    if error is not None:
        raise RuntimeError(
            f"JupyterNotebook ({label}) code error: {error!r}"
        ) from error
    return {k: ns.get(k, v) for k, v in values.items()}


class JupyterNotebook:
    """Inline notebook cell inside the node body, with up to 10 dynamic slots.

    `value` is required; `b`...`j` are optional. The web extension hides the
    optionals by default and reveals the next one whenever the trailing slot
    becomes wired (rgthree-style dynamic IO). All wired slots are exposed in
    the kernel namespace under their slot names; reassigning any of them in
    the cell flows the new value to the matching output slot.

    interactive=True  -> pause; the embedded UI runs cells against the
                        shared kernel, edits to any slot flow downstream on
                        resume.
    interactive=False -> exec the stored `code` once on queue against the
                        incoming slot values. Whatever the code rebinds
                        each name to is its slot's output.
    """

    SLOTS = ("value", "b", "c", "d", "e", "f", "g", "h", "i", "j")
    MODES = ("no kernel", "kernel + pause", "kernel + continue")
    DEFAULT_MODE = "kernel + pause"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (ANY, {}),
                "mode": (list(cls.MODES), {"default": cls.DEFAULT_MODE}),
                # `label` is legacy: hidden in the UI, kept so old workflows
                # still load. The real label is the sanitized node title.
                "label": ("STRING", {"default": ""}),
                "code": ("STRING", {"multiline": True, "default": ""}),
                "session": ("STRING", {"default": "default"}),
            },
            "optional": {name: (ANY, {}) for name in cls.SLOTS[1:]},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = tuple([ANY] * len(SLOTS))
    RETURN_NAMES = SLOTS
    FUNCTION = "run"
    CATEGORY = "debug"
    OUTPUT_NODE = True   # always execute, even with no downstream consumer

    @classmethod
    def IS_CHANGED(cls, value, mode, label, code, session="default",
                   b=None, c=None, d=None, e=None, f=None, g=None,
                   h=None, i=None, j=None, unique_id=None,
                   extra_pnginfo=None):
        return time.time()

    def run(self, value, mode, label, code, session="default",
            b=None, c=None, d=None, e=None, f=None, g=None,
            h=None, i=None, j=None, unique_id=None, extra_pnginfo=None):
        label = _resolve_label_from_workflow(
            extra_pnginfo, unique_id, default="notebook"
        )
        inputs = dict(zip(self.SLOTS,
                          [value, b, c, d, e, f, g, h, i, j]))

        if mode == "kernel + pause":
            out = _do_pause_multi(inputs, label, print_attach_block=False,
                                  session=session)
        elif mode == "kernel + continue":
            if not code or not code.strip():
                return tuple(inputs[k] for k in self.SLOTS)
            out = _exec_noninteractive_multi(
                inputs, label, code, unique_id=unique_id, session=session,
            )
        else:   # "no kernel"
            if not code or not code.strip():
                return tuple(inputs[k] for k in self.SLOTS)
            out = _exec_no_kernel(inputs, label, code, unique_id=unique_id)

        return tuple(out[k] for k in self.SLOTS)


NODE_CLASS_MAPPINGS = {
    "JupyterBreakpoint": JupyterBreakpoint,
    "JupyterNotebook": JupyterNotebook,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "JupyterBreakpoint": "Jupyter Breakpoint",
    "JupyterNotebook": "Jupyter Notebook",
}


# ---------------------------------------------------------------------------
# HTTP routes on ComfyUI's aiohttp server. The in-node UI hits these.
# Registered at import time; wrapped in try/except so missing-server (e.g.
# the smoketest, or imports during tooling) doesn't break the module.
# ---------------------------------------------------------------------------
try:
    import server as _comfy_server  # type: ignore
    from aiohttp import web as _aiohttp_web  # type: ignore

    _routes = _comfy_server.PromptServer.instance.routes

    @_routes.post("/compyter/execute")
    async def _execute_endpoint(request):  # type: ignore[misc]
        try:
            body = await request.json()
        except Exception:
            body = {}
        code = body.get("code", "") if isinstance(body, dict) else ""
        session = (body.get("session") if isinstance(body, dict) else None) or "default"
        label = (body.get("label") if isinstance(body, dict) else None) or ""
        loop = asyncio.get_event_loop()

        def _do():
            _ensure_kernel()
            _activate_session(session)
            if label:
                _kernel_state["user_ns"]["label"] = label
            result = _execute_in_kernel(code)
            _save_active_session()
            return result

        try:
            result = await loop.run_in_executor(None, _do)
            return _aiohttp_web.json_response(result)
        except Exception as e:
            return _aiohttp_web.json_response(
                {"ok": False, "error": repr(e)}, status=500
            )

    @_routes.post("/compyter/resume")
    async def _resume_endpoint(request):  # type: ignore[misc]
        pause_event: threading.Event = _kernel_state["pause_event"]
        pause_event.set()
        return _aiohttp_web.json_response({"ok": True})

    @_routes.get("/compyter/status")
    async def _status_endpoint(request):  # type: ignore[misc]
        return _aiohttp_web.json_response({
            "started": bool(_kernel_state.get("started")),
            "paused": bool(_kernel_state.get("paused")),
            "label": _kernel_state.get("paused_label"),
            "connection_file": _kernel_state.get("connection_file"),
        })

    @_routes.get("/compyter/outputs")
    async def _outputs_endpoint(request):  # type: ignore[misc]
        node_id = request.query.get("node_id", "")
        buf = _kernel_state.get("last_outputs") or {}
        outs = buf.pop(node_id, None)
        return _aiohttp_web.json_response({"outputs": outs or []})
except Exception:
    log.debug("Compyter: HTTP routes not registered (ComfyUI server unavailable).")
