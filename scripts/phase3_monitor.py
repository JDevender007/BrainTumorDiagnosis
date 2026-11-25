"""Continuous Phase 3 training monitor - no prompts, no model turns.

Samples every N seconds and appends one row per run per epoch transition to
``results/phase3/monitor_log.csv``, while keeping
``results/phase3/monitor_latest.txt`` current (rewritten each cycle).

It also records events (rotation to the next queue entry, run finished, failure
patterns in the run log) in ``results/phase3/monitor_events.log``.

It never touches a training process: read-only. It exits on its own once no
trainer has been alive for a few consecutive cycles.

Usage:
    python -X utf8 scripts/phase3_monitor.py
    python -X utf8 scripts/phase3_monitor.py --interval 30
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

RUNS_DIR = Path("results/phase3/runs")
PHASE3 = Path("results/phase3")
CSV_PATH = PHASE3 / "monitor_log.csv"
LATEST = PHASE3 / "monitor_latest.txt"
EVENTS = PHASE3 / "monitor_events.log"

FIELDS = [
    "ts_utc", "run", "epoch", "epochs_total", "best_f1", "best_epoch",
    "epoch_s", "eta_s", "gpu_util", "gpu_mem_mib", "commit_gb", "commit_limit_gb",
]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def commit_gb() -> tuple[float, float]:
    """(committed, limit) in GB from GlobalMemoryStatusEx."""
    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return (0.0, 0.0)
    used = (st.ullTotalPageFile - st.ullAvailPageFile) / 1024**3
    limit = st.ullTotalPageFile / 1024**3
    return (round(used, 1), round(limit, 1))


def gpu() -> tuple[int, int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        util, mem = [p.strip() for p in out.split(",")[:2]]
        return (int(float(util)), int(float(mem)))
    except Exception:
        return (-1, -1)


def short_label(name: str) -> str:
    """tamnet_clean_s43+mc -> 's43 +mc'; densenet121_clean_s43 -> 'densenet121 s43'."""
    n = name
    if n.startswith("tamnet_clean_s"):
        rest = n[len("tamnet_clean_"):]          # s43+mc
        seed, _, suffix = rest.partition("+")
        return f"{seed} +{suffix}" if suffix else seed
    if n.startswith("shared_unet_brisc_s"):
        seed = n.split("_s", 1)[1].split("+")[0]
        return f"shared_unet s{seed}"
    if n.startswith("efficientnet_v2_s_clean_s"):
        return f"effnet s{n.rsplit('_s', 1)[1]}"
    if n.startswith("swin_t_clean_s"):
        return f"swin s{n.rsplit('_s', 1)[1]}"
    if "_clean_s" in n:
        model, seed = n.split("_clean_s", 1)
        return f"{model} s{seed}"
    return n


def load_queues() -> list[tuple[str, list[str]]]:
    """[(queue file, [run names in order])] for every queue that still has work."""
    out = []
    for q in sorted(PHASE3.glob("queue_stage*.json")):
        data = read_json(q)
        if not isinstance(data, list):
            continue
        names = [e.get("name") for e in data if e.get("name")]
        if names:
            out.append((q.name, names))
    return out


def next_entry(name: str, queues: list[tuple[str, list[str]]]) -> str | None:
    """The entry that will start after ``name`` in its own queue."""
    for _, names in queues:
        if name not in names:
            continue
        for cand in names[names.index(name) + 1:]:
            if not (RUNS_DIR / cand / "training_result.json").is_file():
                return cand
    return None


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def _run_name_from_cmd(cmd: list[str]) -> str | None:
    """Derive the run name exactly as phase3_train.py does when --out is absent.

    The queue never passes --out, so name detection must replicate the
    trainer's model/protocol/seed + suffix rules, otherwise the monitor
    reports "0 trainers" while training is live.
    """
    opts: dict[str, str] = {}
    flags: set[str] = set()
    i = 0
    while i < len(cmd):
        tok = cmd[i]
        if tok.startswith("--") and i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
            opts[tok[2:]] = cmd[i + 1]
            i += 2
        elif tok.startswith("--"):
            flags.add(tok[2:])
            i += 1
        else:
            i += 1

    if "out" in opts:
        return Path(opts["out"].replace("\\", "/")).name
    if "model" not in opts:
        return None
    model = opts["model"]
    protocol = opts.get("protocol", "clean")
    seed = opts.get("seed", "42")
    suffix = ""
    if model in ("tamnet", "shared_unet"):
        if "channel-attention" in flags:
            suffix += "+mc"
        if "no-attention-fusion" in flags:
            suffix += "+concat"
        if "no-seg-head" in flags:
            suffix += "-seg"
        if "no-uncertainty-head" in flags:
            suffix += "-unc"
        if opts.get("seg-decoder") == "multiscale":
            suffix += "+msdec"
        if opts.get("fusion-mode") == "t":
            suffix += "-tonly"
        try:
            if float(opts.get("seg-weight", "0.5")) != 0.5:
                suffix += f"+ls{float(opts['seg-weight']):g}"
        except ValueError:
            pass
        try:
            if int(opts.get("seg-classes", "4")) != 4:
                suffix += f"+sc{int(opts['seg-classes'])}"
        except ValueError:
            pass
    return f"{model}_{protocol}_s{seed}{suffix}"


def live_runs() -> dict[str, str]:
    """run name -> log path, for trainers that are actually alive."""
    out: dict[str, str] = {}
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if (p.info["name"] or "").lower() not in ("python.exe", "python3.exe", "pythonw.exe"):
                continue
            cmd = p.info["cmdline"] or []
            if not any("phase3_train" in t for t in cmd):
                continue
            name = _run_name_from_cmd([str(t) for t in cmd])
            if name:
                out[name] = str(p.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def log_event(msg: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with EVENTS.open("a", encoding="utf-8") as f:
        f.write(f"{stamp}  {msg}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--max-idle-cycles", type=int, default=4,
                    help="exit after this many consecutive cycles with no trainer")
    args = ap.parse_args()

    PHASE3.mkdir(parents=True, exist_ok=True)
    new_csv = not CSV_PATH.exists()
    csv_f = CSV_PATH.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_f, fieldnames=FIELDS)
    if new_csv:
        writer.writeheader()

    seen: dict[str, int] = {}          # run -> last logged epoch
    finished: set[str] = set()
    idle = 0
    log_event("monitor started")
    print(f"monitor: sampling every {args.interval:.0f}s -> {CSV_PATH}", flush=True)

    try:
        while True:
            runs = live_runs()
            if not runs:
                idle += 1
                if idle >= args.max_idle_cycles:
                    log_event("monitor exiting: no trainer alive")
                    print("monitor: no trainer alive, exiting", flush=True)
                    break
            else:
                idle = 0

            g_util, g_mem = gpu()
            c_used, c_limit = commit_gb()
            rows = []
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

            for name in sorted(runs):
                prog = read_json(RUNS_DIR / name / "progress.json")
                if not prog:
                    continue
                epoch = prog.get("epoch")
                if seen.get(name) != epoch:
                    if name in seen:
                        log_event(f"{name}: epoch {seen[name]} -> {epoch} "
                                  f"(best {prog.get('best_val_f1'):.4f}@{prog.get('best_epoch')})")
                    else:
                        log_event(f"{name}: first seen at epoch {epoch}")
                    seen[name] = epoch
                    writer.writerow({
                        "ts_utc": ts, "run": name, "epoch": epoch,
                        "epochs_total": prog.get("epochs_total"),
                        "best_f1": prog.get("best_val_f1"),
                        "best_epoch": prog.get("best_epoch"),
                        "epoch_s": round(prog.get("epoch_time_s") or 0, 1),
                        "eta_s": round(prog.get("eta_s") or 0, 1),
                        "gpu_util": g_util, "gpu_mem_mib": g_mem,
                        "commit_gb": c_used, "commit_limit_gb": c_limit,
                    })
                    csv_f.flush()
                rows.append((name, prog))

            # detect completions
            for d in RUNS_DIR.iterdir():
                if not d.is_dir() or d.name in finished:
                    continue
                if (d / "training_result.json").is_file() and d.name not in runs:
                    res = read_json(d / "training_result.json") or {}
                    finished.add(d.name)
                    log_event(f"DONE {d.name}: best {res.get('best_val_f1_weighted')} "
                              f"@ ep{res.get('best_epoch')}")

            queues = load_queues()
            lines = []
            soonest = None            # (eta_min, name, prog)
            for name, prog in rows:
                epoch = prog.get("epoch") or 0
                total = prog.get("epochs_total") or 30
                if prog.get("best_epoch") == epoch:
                    best_txt = f"{prog.get('best_val_f1'):.4f}"
                else:
                    best_txt = f"best {prog.get('best_val_f1'):.4f}@{prog.get('best_epoch')}"
                try:
                    age = time.time() - (RUNS_DIR / name / "progress.json").stat().st_mtime
                except OSError:
                    age = 0
                if age > 150 and epoch < total:
                    lines.append(f"- {short_label(name)}: ep{epoch + 1} in progress — {best_txt}")
                else:
                    eta_min = int(round((prog.get("eta_s") or 0) / 60))
                    lines.append(f"- {short_label(name)}: ep{epoch}/{total} — "
                                 f"{best_txt} · ETA {eta_min} min")
                    if soonest is None or eta_min < soonest[0]:
                        soonest = (eta_min, name, prog)

            lines.append(f"- commit {c_used}/{c_limit} ✓")

            note = ""
            if soonest:
                eta_min, sname, sprog = soonest
                nxt = next_entry(sname, queues)
                tail = (f"→ {short_label(nxt)} starts" if nxt else "→ queue finished")
                note = (f"Next: {short_label(sname)} finishes ep{sprog.get('epochs_total')} "
                        f"in ~{eta_min} min {tail}.")
            lines += ["", note, "",
                      f"GPU {g_util}% / {g_mem} MiB · {len(runs)} trainers · "
                      f"{len(finished)} completed runs on disk · {ts}"]
            LATEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log_event("monitor stopped by user")
    finally:
        csv_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
