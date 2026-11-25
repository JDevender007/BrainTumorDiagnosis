#!/usr/bin/env python3
"""After reboot: verify the commit ceiling, then start the next-3 queue runners.

Safety rules:
- Fails safe: if the commit ceiling is < 40 GB (pagefile did not materialize),
  starts NOTHING and records why.
- Starts nothing if any phase3 queue/trainer is already running (no duplicates).
- Skips queues whose entries all have training_result.json; if ALL are complete,
  deletes the scheduled task (self-cleanup) and starts nothing.
- Launches queue_next3_{a,b,c}.json concurrently, each with its own status file
  and log. Detached children survive this script exiting.

Writes results/phase3/after_reboot_start.json. Deletes nothing else.
"""
import ctypes
import datetime
import json
import os
import subprocess
import sys
import time

ROOT = r"D:\BrainTumorDiagnosis"
PY = sys.executable
LOGDIR = os.path.join(ROOT, "results", "phase3", "logs")
OUT = os.path.join(ROOT, "results", "phase3", "after_reboot_start.json")
TASK_NAME = "TAMNet_ResumeNext3"
TAGS = ("a", "b", "c")
MIN_CEILING_GB = 40.0
# Cold-start stagger: launching all 3 queues at once triple-spiked the
# commit charge (2026-11-25 13:29 IST: s45 CUDA OOM, s43 bad allocation).
# Launch one queue, then wait for headroom before the next.
STAGGER_SECONDS = 120
MIN_AVAIL_GB = 16.0
HEADROOM_TIMEOUT_S = 900


def memstatus():
    class MI(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    m = MI()
    m.dwLength = ctypes.sizeof(MI)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return {
        "ceiling_gb": round(m.ullTotalPageFile / 2**30, 2),
        "avail_gb": round(m.ullAvailPageFile / 2**30, 2),
        "ram_total_gb": round(m.ullTotalPhys / 2**30, 2),
        "ram_avail_gb": round(m.ullAvailPhys / 2**30, 2),
    }


def live_phase3():
    hits = []
    try:
        import psutil
        me = psutil.Process().pid
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                if p.info["pid"] == me:
                    continue
                cl = " ".join(p.info["cmdline"] or [])
                if "phase3_queue.py" in cl or "phase3_train.py" in cl:
                    hits.append({"pid": p.info["pid"], "cmd": cl[:160]})
            except Exception:
                pass
    except ImportError:
        pass
    return hits


def queue_state(tag):
    """Return (all_done, entries) for one queue file."""
    qpath = os.path.join(ROOT, "results", "phase3", f"queue_next3_{tag}.json")
    try:
        with open(qpath, encoding="utf-8") as f:
            entries = json.load(f)
    except (OSError, ValueError) as e:
        return None, f"queue unreadable: {e}"
    done = []
    for e in entries:
        name = e["name"]
        rj = os.path.join(ROOT, "results", "phase3", "runs", name, "training_result.json")
        done.append(os.path.exists(rj))
    return all(done), entries


def launch(tag):
    queue = os.path.join(ROOT, "results", "phase3", f"queue_next3_{tag}.json")
    status = os.path.join(ROOT, "results", "phase3", f"queue_status_next3_{tag}.json")
    log = os.path.join(LOGDIR, f"queue_next3_{tag}.log")
    os.makedirs(LOGDIR, exist_ok=True)
    with open(log, "ab") as fh:
        proc = subprocess.Popen(
            [PY, "-X", "utf8",
             os.path.join(ROOT, "scripts", "phase3_queue.py"),
             "--queue", queue, "--status-file", status],
            cwd=ROOT,
            stdout=fh,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    return proc.pid


def maybe_delete_task():
    try:
        subprocess.run(
            ["schtasks", "/delete", "/tn", TASK_NAME, "/f"],
            capture_output=True, timeout=30,
        )
    except Exception:
        pass


def main():
    result = {
        "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "memory_before": memstatus(),
        "started_queues": [],
        "pids": {},
        "skipped": [],
    }
    try:
        ceiling = result["memory_before"]["ceiling_gb"]
        if ceiling < MIN_CEILING_GB:
            result["started"] = False
            result["reason"] = (
                f"commit ceiling {ceiling} GB < {MIN_CEILING_GB} GB - pagefile missing; "
                "run scripts/create_pagefile.py elevated and reboot again"
            )
            _write(result)
            return 1

        live = live_phase3()
        if live:
            result["started"] = False
            result["reason"] = "phase3 already running - no duplicates started"
            result["live"] = live
            _write(result)
            return 0

        pending = []
        for tag in TAGS:
            done, info = queue_state(tag)
            if done is True:
                result["skipped"].append({"tag": tag, "reason": "all entries complete"})
            elif done is False:
                pending.append(tag)
            else:
                result["started"] = False
                result["reason"] = info
                _write(result)
                return 1

        if not pending:
            result["started"] = False
            result["reason"] = "all 3 queues complete"
            maybe_delete_task()
            _write(result)
            return 0

        for i, tag in enumerate(pending):
            if i > 0:
                waited = 0
                while memstatus()["avail_gb"] < MIN_AVAIL_GB and waited < HEADROOM_TIMEOUT_S:
                    time.sleep(30)
                    waited += 30
                time.sleep(STAGGER_SECONDS)
            pid = launch(tag)
            result["pids"][tag] = pid
            result["started_queues"].append(tag)
            _write(result)  # incremental: survive a crash mid-launch
        result["started"] = True
        result["reason"] = f"launched {len(pending)} queue runner(s)"
        result["memory_after_launch"] = memstatus()
    except Exception as e:
        result["started"] = False
        result["reason"] = f"{type(e).__name__}: {e}"
    _write(result)
    return 0 if result.get("started") else 1


def _write(result):
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
