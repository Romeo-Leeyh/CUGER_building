import argparse
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run test.py in parallel with worker-based sharding and CPU affinity."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes to launch.",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=Path("test.py"),
        help="Path to worker script (default: test.py).",
    )
    parser.add_argument(
        "--lod",
        type=str,
        default="precise",
        choices=["precise", "medium", "low"],
        help="LOD forwarded to each worker.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch workers.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print worker commands without running them.",
    )
    return parser.parse_args()


def get_available_cores():
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    cpu_count = os.cpu_count() or 1
    return list(range(cpu_count))


def make_affinity_setter(core_id):
    def _set_affinity():
        os.sched_setaffinity(0, {core_id})

    return _set_affinity


def terminate_processes(processes):
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()

    for proc in processes:
        if proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def reader_thread(proc, state):
    """Consume worker output, tracking latest line and parsed progress counters."""
    if proc.stdout is None:
        return

    progress_re = re.compile(
        r"^progress\s+worker=(?P<worker>\d+)\s+"
        r"completed=(?P<completed>\d+)/(?P<assigned>\d+)\s+"
        r"skipped=(?P<skipped>\d+)\s+processed=(?P<processed>\d+)\s+"
        r"failed=(?P<failed>\d+)"
    )
    summary_re = re.compile(
        r"assigned=(?P<assigned>\d+)\s+"
        r"already_processed=(?P<skipped>\d+)\s+"
        r"processed_now=(?P<processed>\d+)\s+"
        r"failed=(?P<failed>\d+)"
    )

    for line in proc.stdout:
        text = line.strip()
        if text:
            state["last_line"] = text

            progress_match = progress_re.match(text)
            if progress_match:
                state["completed_cases"] = int(progress_match.group("completed"))
                state["assigned_total"] = int(progress_match.group("assigned"))
                state["skipped_cases"] = int(progress_match.group("skipped"))
                state["processed_cases"] = int(progress_match.group("processed"))
                state["failed_cases"] = int(progress_match.group("failed"))
                continue

            summary_match = summary_re.search(text)
            if summary_match:
                assigned = int(summary_match.group("assigned"))
                skipped = int(summary_match.group("skipped"))
                processed = int(summary_match.group("processed"))
                failed = int(summary_match.group("failed"))
                state["assigned_total"] = assigned
                state["skipped_cases"] = skipped
                state["processed_cases"] = processed
                state["failed_cases"] = failed
                state["completed_cases"] = skipped + processed + failed


def render_statuses(states, redraw, previous_line_count):
    total_workers = len(states)
    completed_workers = sum(1 for state in states if state["finished"])
    total_assigned = sum(state.get("assigned_total", 0) for state in states)
    total_completed_cases = sum(state.get("completed_cases", 0) for state in states)

    header = f"workers {completed_workers}/{total_workers}"
    if total_assigned > 0:
        header += f" | cases {total_completed_cases}/{total_assigned}"

    lines = []
    lines.append(header)
    for state in states:
        assigned_total = state.get("assigned_total", 0)
        completed_cases = state.get("completed_cases", 0)
        skipped_cases = state.get("skipped_cases", 0)
        processed_cases = state.get("processed_cases", 0)
        failed_cases = state.get("failed_cases", 0)

        case_bits = ""
        if assigned_total > 0:
            case_bits = (
                f" cases={completed_cases}/{assigned_total} "
                f"(skip={skipped_cases}, ok={processed_cases}, fail={failed_cases})"
            )

        line = (
            f"worker={state['worker_index']} core={state['core_id']} "
            f"status={state['status']}{case_bits} {state['message']}"
        ).rstrip()
        lines.append(line)

    can_redraw = redraw and sys.stdout.isatty() and previous_line_count > 0
    if can_redraw:
        sys.stdout.write(f"\x1b[{previous_line_count}F")

    for line in lines:
        sys.stdout.write("\x1b[2K" + line + "\n")
    sys.stdout.flush()
    return len(lines)


def main():
    args = parse_args()

    if args.workers < 1:
        print(f"Error: --workers must be >= 1, got {args.workers}")
        return 2

    script_path = args.script.resolve()
    if not script_path.exists():
        print(f"Error: script not found: {script_path}")
        return 2

    available_cores = get_available_cores()
    if not available_cores:
        print("Error: no CPU cores available")
        return 2

    if args.dry_run:
        print(f"Launching {args.workers} workers for {script_path.name}")

    worker_states = []
    processes = []
    readers = []
    for worker_index in range(args.workers):
        core_id = available_cores[worker_index % len(available_cores)]
        cmd = [
            args.python,
            str(script_path),
            "--workers",
            str(args.workers),
            "--worker-index",
            str(worker_index),
            "--lod",
            args.lod,
        ]

        if args.dry_run:
            print(
                f"worker={worker_index} core={core_id} --workers {args.workers} "
                f"--worker-index {worker_index} --lod {args.lod}"
            )
            continue

        state = {
            "worker_index": worker_index,
            "core_id": core_id,
            "status": "RUNNING",
            "message": "",
            "last_line": "",
            "finished": False,
            "assigned_total": 0,
            "completed_cases": 0,
            "skipped_cases": 0,
            "processed_cases": 0,
            "failed_cases": 0,
        }
        worker_states.append(state)

        preexec = None
        if hasattr(os, "sched_setaffinity"):
            preexec = make_affinity_setter(core_id)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(script_path.parent),
            env=env,
            preexec_fn=preexec,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes.append(proc)

        reader = threading.Thread(target=reader_thread, args=(proc, state), daemon=True)
        reader.start()
        readers.append(reader)

    if args.dry_run:
        print("Dry run complete.")
        return 0

    rendered_line_count = render_statuses(
        worker_states,
        redraw=False,
        previous_line_count=0,
    )

    try:
        completed = 0
        while completed < len(processes):
            changed = False
            refresh_due = False
            for idx, proc in enumerate(processes):
                state = worker_states[idx]
                if state["finished"]:
                    continue

                rc = proc.poll()
                if rc is None:
                    if state["last_line"] != state["message"]:
                        state["message"] = state["last_line"]
                        refresh_due = True
                    continue

                state["finished"] = True
                state["status"] = "DONE" if rc == 0 else f"FAILED({rc})"
                state["message"] = state["last_line"]
                completed += 1
                changed = True

            if changed or refresh_due:
                rendered_line_count = render_statuses(
                    worker_states,
                    redraw=True,
                    previous_line_count=rendered_line_count,
                )

            if completed < len(processes):
                time.sleep(0.2)

        for reader in readers:
            reader.join(timeout=1)
    except KeyboardInterrupt:
        print("\nInterrupted. Stopping workers...")
        terminate_processes(processes)
        return 130

    failed = [
        state["worker_index"]
        for idx, state in enumerate(worker_states)
        if processes[idx].returncode != 0
    ]
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
