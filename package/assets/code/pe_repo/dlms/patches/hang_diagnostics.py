"""Opt-in live Python stack dumps for distributed-training canaries.

Import this module as a custom register path, then send ``SIGRTMIN+1`` (normally
signal 35 on Linux) to an individual training rank. Unlike SIGUSR1/SIGUSR2,
this signal is not owned by Swift's graceful-shutdown manager. The handler is
diagnostic only and has no background thread or steady-state overhead.
"""

from __future__ import annotations

import faulthandler
import os
import signal
import sys


_SIGNAL = signal.SIGRTMIN + 1
faulthandler.register(_SIGNAL, file=sys.stderr, all_threads=True, chain=False)
print(
    f"[dlms] hang diagnostics armed: pid={os.getpid()} rank={os.getenv('RANK', '?')} "
    f"signal={int(_SIGNAL)}",
    flush=True,
)
