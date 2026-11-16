"""
BrainTumorDiagnosis
Training progress monitor.

Shows live training progress (percent complete) while the model
is training in the background:

    - overall progress % and progress bar
    - current epoch % (batch-level when the trainer logs batches,
      otherwise estimated from epoch timing)
    - epoch history: train loss / validation dice
    - ETA based on average epoch duration
    - the python processes involved (PID, memory)
    - GPU utilization and VRAM (when available)

Usage (from the project root):

    python -m scripts.training_progress
    python -m scripts.training_progress --refresh 5
    python -m scripts.training_progress --once
    python -m scripts.training_progress --log outputs/segmentation_unet3d/training.log --epochs 30
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LOG = PROJECT_ROOT / "outputs" / "segmentation_unet3d" / "training.log"

DEFAULT_PID_FILE = PROJECT_ROOT / "outputs" / "segmentation_unet3d" / "training.pid"

TIMESTAMP_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
)

PIPELINE_READY_RE = re.compile(r"epochs=(?P<epochs>\d+)")

START_EPOCH_RE = re.compile(r"Start epoch (?P<epoch>\d+)/(?P<total>\d+)")

TRAIN_DONE_RE = re.compile(
    r"Train epoch (?P<epoch>\d+)/(?P<total>\d+) \| loss (?P<loss>\d+\.\d+)"
)

VAL_DONE_RE = re.compile(
    r"Val epoch (?P<epoch>\d+)/(?P<total>\d+) \| loss \d+\.\d+ "
    r"\| mean foreground dice (?P<dice>\d+\.\d+)"
)

BATCH_PROGRESS_RE = re.compile(
    r"Train epoch (?P<epoch>\d+)/(?P<total>\d+) \| "
    r"batch (?P<batch>\d+)/(?P<batches>\d+) \| "
    r"(?P<pct>\d+(?:\.\d+)?)% \| loss (?P<loss>\d+\.\d+)"
)

EARLY_STOP_RE = re.compile(r"Early stopping after (?P<epochs>\d+) epochs")


class TrainingState:
    """Parsed state of a running training job."""

    def __init__(self) -> None:
        self.total_epochs: int | None = None
        self.pipeline_total_epochs: int | None = None
        self.epoch_starts: dict[int, datetime.datetime] = {}
        self.train_completions: dict[int, float] = {}
        self.val_completions: dict[int, float] = {}
        self.batch_progress: dict[int, tuple[int, int, float]] = {}
        self.early_stopped_at: int | None = None
        self.last_line_ts: datetime.datetime | None = None

    @property
    def completed_epochs(self) -> int:
        return len(self.val_completions)

    @property
    def current_epoch(self) -> int | None:
        if self.epoch_starts:
            return max(self.epoch_starts)

        if self.completed_epochs > 0:
            return self.completed_epochs + 1

        return 1

    @property
    def finished(self) -> bool:
        if self.early_stopped_at is not None:
            return True
        if self.total_epochs is None:
            return False
        return self.completed_epochs >= self.total_epochs

    def best_dice(self) -> float | None:
        if not self.val_completions:
            return None
        return max(self.val_completions.values())

    def average_epoch_seconds(self) -> float | None:
        timestamps = sorted(self.epoch_starts.values())

        if len(timestamps) >= 2:
            diffs = [
                (b - a).total_seconds()
                for a, b in zip(timestamps, timestamps[1:])
            ]
            return sum(diffs) / len(diffs)

        if len(self.train_completions) >= 2:
            diffs = list(self.train_completions.values())

            spans = [
                b - a for a, b in zip(diffs, diffs[1:])
            ]

            if spans:
                return sum(spans) / len(spans)

        return None


def parse_log(log_path: Path, state: TrainingState) -> None:
    """Parse the training log file into a TrainingState."""

    if not log_path.is_file():
        return

    with log_path.open("r", encoding="utf-8", errors="replace") as file:
        content = file.read()

    for line in content.splitlines():
        match = TIMESTAMP_RE.match(line)

        if match is None:
            continue

        timestamp = datetime.datetime.strptime(
            match.group("ts"),
            "%Y-%m-%d %H:%M:%S,%f",
        )

        state.last_line_ts = timestamp

        ready = PIPELINE_READY_RE.search(line)

        if ready is not None:
            state.pipeline_total_epochs = int(ready.group("epochs"))

        start = START_EPOCH_RE.search(line)

        if start is not None:
            epoch = int(start.group("epoch"))

            state.total_epochs = state.total_epochs or int(
                start.group("total")
            )

            state.epoch_starts.setdefault(epoch, timestamp)

            continue

        batch = BATCH_PROGRESS_RE.search(line)

        if batch is not None:
            epoch = int(batch.group("epoch"))

            state.total_epochs = state.total_epochs or int(
                batch.group("total")
            )

            state.batch_progress[epoch] = (
                int(batch.group("batch")),
                int(batch.group("batches")),
                float(batch.group("pct")),
            )

            continue

        train_done = TRAIN_DONE_RE.search(line)

        if train_done is not None:
            epoch = int(train_done.group("epoch"))

            state.total_epochs = state.total_epochs or int(
                train_done.group("total")
            )

            state.train_completions[epoch] = (
                timestamp.timestamp()
            )

            continue

        val_done = VAL_DONE_RE.search(line)

        if val_done is not None:
            epoch = int(val_done.group("epoch"))

            state.total_epochs = state.total_epochs or int(
                val_done.group("total")
            )

            state.val_completions[epoch] = float(val_done.group("dice"))

            continue

        early_stop = EARLY_STOP_RE.search(line)

        if early_stop is not None:
            state.early_stopped_at = int(early_stop.group("epochs"))

    if state.pipeline_total_epochs is not None:
        state.total_epochs = state.pipeline_total_epochs


def estimate_overall_progress(
    state: TrainingState,
) -> tuple[float, float]:
    """
    Return (overall_percent, current_epoch_percent).

    Batch-level progress is used when the trainer logs it;
    otherwise the fraction is estimated from the elapsed time
    inside the current epoch relative to the average epoch
    duration.
    """

    if state.total_epochs is None or state.total_epochs <= 0:
        return 0.0, 0.0

    if state.finished:
        return 100.0, 100.0

    completed = state.completed_epochs

    current_epoch = state.current_epoch

    epoch_fraction = 0.0

    if current_epoch is not None:
        batch = state.batch_progress.get(current_epoch)

        if batch is not None:
            epoch_fraction = min(
                1.0,
                max(0.0, batch[2] / 100.0),
            )
        else:
            start_ts = state.epoch_starts.get(current_epoch)

            average = state.average_epoch_seconds()

            if start_ts is not None and average and average > 0:
                elapsed = (
                    datetime.datetime.now() - start_ts
                ).total_seconds()

                epoch_fraction = min(
                    0.99,
                    max(0.0, elapsed / average),
                )

    overall = (completed + epoch_fraction) / state.total_epochs * 100.0

    return overall, epoch_fraction * 100.0


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)

    hours, remainder = divmod(int(seconds), 3600)

    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"

    if minutes > 0:
        return f"{minutes}m {secs:02d}s"

    return f"{secs}s"


def process_info() -> list[str]:
    """Collect python process rows (PID, memory)."""

    rows: list[str] = []

    try:
        if os.name == "nt":
            output = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH", "/FI", "IMAGENAME eq python.exe"],
                capture_output=True,
                text=True,
                timeout=15,
            ).stdout

            task_re = re.compile(
                r'^"([^"]*)","(\d+)","[^"]*","[^"]*","([^"]*)"'
            )

            for line in output.splitlines():
                match = task_re.match(line)

                if match is not None:
                    rows.append(
                        f"  {match.group(1)} PID={match.group(2)} "
                        f"MEM={match.group(3)}"
                    )
        else:
            for proc in Path("/proc").glob("[0-9]*"):
                try:
                    with (proc / "comm").open() as file:
                        command = file.read().strip()

                    if command == "python":
                        with (proc / "status").open() as file:
                            status = file.read()

                        mem_match = re.search(
                            r"VmRSS:\s+(\d+) kB",
                            status,
                        )

                        mem = (
                            f"{int(mem_match.group(1)) // 1024} MB"
                            if mem_match
                            else "?"
                        )

                        rows.append(f"  python PID={proc.name} MEM={mem}")
                except OSError:
                    continue
    except Exception as exc:  # pragma: no cover - best effort
        rows.append(f"  (process info unavailable: {exc})")

    if not rows:
        rows.append("  (no python processes found)")

    return rows


def gpu_info() -> list[str]:
    """Collect GPU utilization and VRAM via nvidia-smi."""

    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout

        rows: list[str] = []

        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]

            if len(parts) == 3:
                rows.append(
                    f"  GPU util={parts[0]}% | VRAM {parts[1]} / {parts[2]} MB"
                )

        if rows:
            return rows

        return ["  (no GPU data)"]
    except Exception:
        return ["  (nvidia-smi unavailable)"]


def render(
    state: TrainingState,
    overall_percent: float,
    epoch_percent: float,
    pid: int | None,
) -> str:
    """Render the progress report block."""

    width = 40

    filled = int(round(overall_percent / 100.0 * width))

    bar = "#" * filled + "-" * (width - filled)

    lines: list[str] = []

    lines.append("=" * 78)
    lines.append("BRAIN TUMOR DIAGNOSIS - TRAINING PROGRESS")
    lines.append("=" * 78)

    if state.total_epochs is None:
        lines.append("No training activity detected yet (no log data).")
        lines.append("=" * 78)
        return "\n".join(lines)

    lines.append(f"  [ {bar} ]  {overall_percent:5.1f}% complete")

    lines.append(
        f"  Epochs        : {state.completed_epochs} / "
        f"{state.total_epochs} completed"
    )

    if state.finished:
        status = "COMPLETED"

        if state.early_stopped_at is not None:
            status = (
                f"EARLY STOPPED after {state.early_stopped_at} "
                "epochs without improvement"
            )

        lines.append(f"  Status        : {status}")
    else:
        current = state.current_epoch or 1

        lines.append(
            f"  Current epoch : {current} / {state.total_epochs} "
            f"({epoch_percent:5.1f}% of this epoch)"
        )

    best = state.best_dice()

    if best is not None:
        lines.append(f"  Best val dice : {best:.4f} (mean foreground)")

    last_epoch = state.completed_epochs

    if last_epoch > 0 and last_epoch in state.val_completions:
        lines.append(
            f"  Last epoch dice: {state.val_completions[last_epoch]:.4f}"
        )

    average = state.average_epoch_seconds()

    if average is not None and not state.finished:
        remaining = (
            state.total_epochs
            - state.completed_epochs
            - (epoch_percent / 100.0)
        )

        lines.append(
            f"  Avg epoch time: {format_duration(average)} "
            f"| ETA: {format_duration(remaining * average)}"
        )

    lines.append("-" * 78)

    if pid is not None:
        lines.append(f"  Training PID  : {pid}")

    lines.append("  Processes     :")
    lines.extend(process_info())

    lines.append("  GPU           :")
    lines.extend(gpu_info())

    lines.append("=" * 78)

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor background training progress.",
    )

    parser.add_argument(
        "--log",
        type=str,
        default=str(DEFAULT_LOG),
        help="Path to the training log file.",
    )

    parser.add_argument(
        "--pid-file",
        type=str,
        default=str(DEFAULT_PID_FILE),
        help="Path to the training PID file.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the total number of epochs.",
    )

    parser.add_argument(
        "--refresh",
        type=float,
        default=10.0,
        help="Refresh interval in seconds (default: 10).",
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Print a single report and exit.",
    )

    return parser


def main() -> int:
    parser = build_parser()

    args = parser.parse_args()

    log_path = Path(args.log)

    pid = None

    pid_file = Path(args.pid_file)

    if pid_file.is_file():
        try:
            pid = int(pid_file.read_text().strip())
        except ValueError:
            pid = None

    state = TrainingState()

    try:
        while True:
            state = TrainingState()

            parse_log(log_path, state)

            if args.epochs is not None:
                state.total_epochs = args.epochs

            overall_percent, epoch_percent = estimate_overall_progress(
                state
            )

            report = render(
                state,
                overall_percent,
                epoch_percent,
                pid,
            )

            if os.name == "nt":
                os.system("cls")
            else:
                os.system("clear")

            print(report)

            if args.once or state.finished:
                return 0

            time.sleep(max(0.5, args.refresh))
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
