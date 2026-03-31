import argparse
import contextlib
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Ensure workspace root is importable when running this file directly.
CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from cuger.__transform import process as ps
import moosas.MoosasPy as Moosas

DEFAULT_INPUT_DIR = Path("cuger/tests/examples")
DEFAULT_OUTPUT_DIR = Path("cuger/tests/examples_results")
DEFAULT_LOD = "medium"
DEFAULT_ENABLE_MINIMAL_CORE = True
DEFAULT_WORKERS = os.cpu_count() or 1


@contextlib.contextmanager
def suppress_output():
    """Silence both Python-level and low-level stdout/stderr in this process."""
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)

    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr


def _collect_geo_files(folder: Path) -> list[tuple[Path, str]]:
    if not folder.is_dir():
        return []

    geo_files: list[tuple[Path, str]] = []
    for geo_filepath in sorted(folder.rglob("*.geo")):
        relative_path = geo_filepath.relative_to(folder)
        modelname = relative_path.with_suffix("").as_posix().replace("/", "_")
        geo_files.append((geo_filepath, modelname))

    return geo_files


def is_file_processed(modelname: str, output_dir: Path, lod: str = "precise") -> bool:
    """Return True when the expected pipeline outputs for a model already exist."""
    paths = ps.get_output_paths(modelname, str(output_dir), lod=lod)
    required_outputs = [
        paths["simplified_geo_path"],
        paths["convex_geo_path"],
        paths["new_geo_path"],
        paths["new_xml_path"],
        paths["new_idf_path"],
        paths["output_graph_path"],
    ]
    return all(Path(path).exists() for path in required_outputs)


def _process_file(
    input_geo_path_str: str,
    output_dir_str: str,
    modelname: str,
    lod: str,
    enable_minimal_core: bool,
) -> tuple[bool, str, str]:
    """Run the transform pipeline for one GEO file."""
    input_geo_path = Path(input_geo_path_str)
    output_dir = Path(output_dir_str)
    paths = ps.get_output_paths(modelname, str(output_dir), lod=lod)

    try:
        ps.simplify_process(
            str(input_geo_path),
            paths["simplified_geo_path"],
            lod=lod,
            enable_minimal_core=enable_minimal_core,
        )

        ps.convex_process(
            paths["simplified_geo_path"],
            paths["convex_geo_path"],
            paths["figure_convex_path"],
        )

        with suppress_output():
            model = Moosas.transform(
                paths["convex_geo_path"],
                solve_overlap=True,
                divided_zones=False,
                break_wall_horizontal=True,
                solve_redundant=True,
                attach_shading=False,
                standardize=True,
                stdout=io.StringIO(),
            )

            Moosas.saveModel(model, paths["new_geo_path"], save_type="geo")
            Moosas.saveModel(model, paths["new_xml_path"], save_type="xml")
            #Moosas.saveModel(model, paths["new_idf_path"], save_type="idf")

        ps.graph_process(
            paths["new_geo_path"],
            paths["new_xml_path"],
            paths["output_graph_path"],
            paths["figure_graph_path"],
        )
    except Exception as exc:
        return False, modelname, str(exc)

    return True, modelname, "ok"


def _format_core_status(active_workers: int, workers: int) -> str:
    labels: list[str] = []
    for idx in range(workers):
        state = "RUN" if idx < active_workers else "IDLE"
        labels.append(f"C{idx + 1}:{state}")
    return " ".join(labels)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch run the CUGER transform pipeline for GEO files."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Folder containing input .geo files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output folder for transformed results.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of worker processes for batch transforms.",
    )
    parser.add_argument(
        "--lod",
        type=str,
        default=DEFAULT_LOD,
        choices=["precise", "medium", "low"],
        help="Level of detail for simplification.",
    )
    parser.add_argument(
        "--enable-minimal-core",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_MINIMAL_CORE,
        help="Inject a minimal core shaft into low/medium simplified geometry.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Print progress every N completed jobs (default: 10).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log output.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if args.workers < 1:
        print("--workers must be >= 1.")
        return 1

    if args.log_interval < 1:
        print("--log-interval must be >= 1.")
        return 1

    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}")
        return 1

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Cannot access --output-dir: {output_dir}")
        print(f"OS error: {exc}")
        return 1

    geo_files = _collect_geo_files(input_dir)
    if not geo_files:
        print(f"No GEO files found in: {input_dir}")
        return 1

    pending_jobs: list[tuple[Path, str]] = []
    skipped = 0
    for geo_path, modelname in geo_files:
        if is_file_processed(modelname, output_dir, lod=args.lod):
            skipped += 1
            continue
        pending_jobs.append((geo_path, modelname))

    total_pending = len(pending_jobs)
    overall_total = skipped + total_pending

    if not args.quiet:
        print(
            f"Total GEO files={len(geo_files)}, Pending jobs={total_pending}, "
            f"Completed before run={skipped}"
        )

    if total_pending == 0:
        print("All jobs already completed. Exiting.")
        return 0

    logical_cpus = os.cpu_count() or 1
    workers = min(args.workers, total_pending, logical_cpus)
    done = 0
    ok = 0
    failed = 0

    job_payloads = [
        (
            str(geo_path),
            str(output_dir),
            modelname,
            args.lod,
            args.enable_minimal_core,
        )
        for geo_path, modelname in pending_jobs
    ]

    job_iter = iter(job_payloads)
    active_futures = set()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for _ in range(workers):
            try:
                active_futures.add(executor.submit(_process_file, *next(job_iter)))
            except StopIteration:
                break

        while active_futures:
            for future in as_completed(active_futures):
                active_futures.remove(future)
                done += 1

                try:
                    success, modelname, detail = future.result()
                except Exception as exc:
                    failed += 1
                    modelname = "<unknown>"
                    detail = str(exc)
                    success = False
                else:
                    if success:
                        ok += 1
                    else:
                        failed += 1

                try:
                    active_futures.add(executor.submit(_process_file, *next(job_iter)))
                except StopIteration:
                    pass

                if not args.quiet and (done % args.log_interval == 0 or done == total_pending):
                    core_status = _format_core_status(len(active_futures), workers)
                    processed = skipped + done
                    print(
                        f"\rProcessed files: {processed}/{overall_total} | "
                        f"Success={ok} Failed={failed} | Core status: {core_status}",
                        end="",
                        flush=True,
                    )

                if not success and not args.quiet:
                    print(f"\nFailed: {modelname} | {detail}", flush=True)

                break

    if not args.quiet:
        print()

    print("\nBatch finished")
    print(f"  Completed before run: {skipped}")
    print(f"  Success in this run: {ok}")
    print(f"  Failed in this run:  {failed}")
    print(f"  Total attempted in this run: {ok + failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
