"""
Standalone smoke test -- exercises the kernel manager without ComfyUI.

  python smoketest.py

Prints the attach block, waits on the same pause_event the node uses, exits
once you call resume() from the connected front-end.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jupyter_breakpoint import (
    _attach_block,
    _ensure_kernel,
    _kernel_state,
    _make_resume,
)


def main():
    _ensure_kernel()

    pause_event = _kernel_state["pause_event"]
    pause_event.clear()

    ns = _kernel_state["user_ns"]
    ns["value"] = {"hello": "world", "now": time.time(), "msg": "smoke test payload"}
    ns["label"] = "smoketest"
    ns["resume"] = _make_resume(pause_event, _kernel_state["orig_stdout"])

    out = _kernel_state["orig_stdout"] or sys.stdout
    out.write(_attach_block(_kernel_state["connection_file"], "smoketest"))
    out.write("[smoketest] waiting for resume() ... Ctrl-C aborts.\n")
    out.flush()

    pause_event.wait()
    out.write("[smoketest] resumed. exiting.\n")
    out.flush()


if __name__ == "__main__":
    main()
