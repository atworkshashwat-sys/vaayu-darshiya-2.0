"""Find what actually blocks /api/status while a job is refreshing.

Runs a readings refresh on a background thread (as the scheduler does), then
times each piece of status_payload() so the blocking call is identified by
measurement rather than by reading the source.
"""
import threading
import time

import live_jobs
import live_store
import live_stream
import tinyfish_service as svc


def timed(label, fn):
    start = time.time()
    try:
        fn()
        return "%-26s %6.2fs  ok" % (label, time.time() - start)
    except Exception as exc:                      # noqa: BLE001
        return "%-26s %6.2fs  EXC %s" % (label, time.time() - start, exc)


# A job running in the background, exactly as the scheduler starts one.
worker = threading.Thread(target=live_jobs.run_job, args=("readings", True),
                          daemon=True)
worker.start()
time.sleep(1.5)                                   # let it get into the network I/O

print("--- while a readings job is in flight ---")
for label, fn in (
    ("job_states", lambda: live_store.job_states()),
    ("usage_snapshot", svc.usage_snapshot),
    ("is_active", svc.is_active),
    ("stream.stats", live_stream.stats),
    ("status_payload", live_jobs.status_payload),
):
    print(timed(label, fn), flush=True)

worker.join(timeout=600)
print("--- after the job finished ---")
print(timed("status_payload", live_jobs.status_payload))
