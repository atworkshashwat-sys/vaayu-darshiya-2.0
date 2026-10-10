"""Reproduce a hang in /api/status and print every thread's stack.

`/health` answers while `/api/status` does not, so the difference between the
two is where to look: health_payload() reads job_state, status_payload() adds
usage_snapshot() (now touching SQLite under the usage lock) and the stream
stats. faulthandler dumps all thread stacks if the call does not return, which
names the lock rather than leaving us guessing.
"""
import faulthandler
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import tinyfish_service as svc
import live_jobs
import live_store

DB = os.path.join(tempfile.gettempdir(), "diag_status_hang.sqlite")
if os.path.exists(DB):
    os.remove(DB)
live_store.reset(DB)

# Mirror what api.py does at startup.
svc.set_usage_sink(live_store.record_usage)
svc.set_usage_baseline(live_store.usage_total_for_day)

# Background writer, so the state matches a server mid-refresh rather than an
# idle one: the sink is what runs after every TinyFish call.
stop = threading.Event()


def churn():
    i = 0
    while not stop.is_set():
        live_store.record_usage({
            "at": "2026-10-10T20:00:00+05:30", "type": "fetch",
            "key": "key_1", "success": 1, "error": None,
            "rotated": 0, "day": svc.usage_day()})
        i += 1
        time.sleep(0.001)


threading.Thread(target=churn, daemon=True).start()

faulthandler.dump_traceback_later(25, exit=True)
t0 = time.time()
payload = live_jobs.status_payload()
elapsed = time.time() - t0

stop.set()
faulthandler.cancel_dump_traceback_later()
print(f"OK in {elapsed:.2f}s used={payload['tinyfish']['used']} "
      f"cap={payload['tinyfish']['cap']}")
sys.exit(0)
