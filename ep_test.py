"""Batch-run EnergyPlus simulations and export SQL outputs to CSV.
"""

import argparse
import csv
import os
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from cuger.__simulate.simulate import locate_eplus, run_single_simulation  
from cuger.__simulate.sqlread import SQLReader  


DEFAULT_IDF_DIR = Path(r"\\166.111.40.8\temp\lyh\SRT\All_DATA\idf")
DEFAULT_EPW_DIR = Path(r"\\166.111.40.8\temp\lyh\SRT\WEATHER")
DEFAULT_CSV_DIR = Path(r"\\166.111.40.8\temp\lyh\SRT\All_DATA\data")
DEFAULT_WORKERS = 8

_WORKER_EPLUS_READY = False


def _collect_files(folder: Path, suffixes: tuple[str, ...]) -> list[Path]:
	if not folder.is_dir():
		return []
	return sorted(
		p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in suffixes
	)


def _build_tag(file_path: Path, root_dir: Path) -> str:
	return file_path.relative_to(root_dir).with_suffix("").as_posix().replace("/", "_")


def _valid_sql(success: bool, sql_path: str | None) -> bool:
	if not success or not sql_path:
		return False
	return Path(sql_path).is_file()


def _sample_files(files: list[Path], ratio: float, rng: random.Random) -> list[Path]:
	if ratio >= 1.0:
		return files
	count = max(1, int(round(len(files) * ratio)))
	count = min(count, len(files))
	return sorted(rng.sample(files, count))


def _format_core_status(active_workers: int, workers: int) -> str:
	labels: list[str] = []
	for idx in range(workers):
		state = "RUN" if idx < active_workers else "IDLE"
		labels.append(f"C{idx + 1}:{state}")
	return " ".join(labels)


def _build_jobs(
    idf_files: list[Path],
    weather_files: list[Path],
    rng: random.Random,
) -> list[tuple[Path, Path]]:
	k = max(1, len(weather_files))
	jobs: list[tuple[Path, Path]] = []
	for idf_path in idf_files:
		sampled_weather = rng.sample(weather_files, k)
		for weather_path in sampled_weather:
			jobs.append((idf_path, weather_path))
	return jobs


def _write_job_pairs_csv(
	jobs: list[tuple[Path, Path]],
	idf_dir: Path,
	epw_dir: Path,
	csv_dir: Path,
) -> Path:
	out_csv = csv_dir / "job_pairs.csv"
	with out_csv.open("w", newline="", encoding="utf-8") as fp:
		writer = csv.writer(fp)
		writer.writerow(["job_index", "idf_path", "weather_path", "idf_tag", "weather_tag", "job_tag"])
		for idx, (idf_path, weather_path) in enumerate(jobs, start=1):
			idf_tag = _build_tag(idf_path, idf_dir)
			weather_tag = _build_tag(weather_path, epw_dir)
			job_tag = f"{idf_tag}__{weather_tag}"
			writer.writerow([idx, str(idf_path), str(weather_path), idf_tag, weather_tag, job_tag])
	return out_csv


def _ensure_worker_eplus(eplus_root_str: str | None) -> None:
	global _WORKER_EPLUS_READY
	if _WORKER_EPLUS_READY:
		return
	locate_eplus(eplus_root_str or None)
	_WORKER_EPLUS_READY = True


def _run_job(
	idf_path_str: str,
	weather_path_str: str,
	idf_dir_str: str,
	epw_dir_str: str,
	csv_dir_str: str,
	eplus_root_str: str,
	quiet: bool,
) -> tuple[bool, str, str]:
	idf_path = Path(idf_path_str)
	weather_path = Path(weather_path_str)
	idf_dir = Path(idf_dir_str)
	epw_dir = Path(epw_dir_str)
	csv_dir = Path(csv_dir_str)

	idf_tag = _build_tag(idf_path, idf_dir)
	weather_tag = _build_tag(weather_path, epw_dir)
	tag = f"{idf_tag}__{weather_tag}"

	try:
		_ensure_worker_eplus(eplus_root_str)
	except Exception as exc:
		return False, tag, f"EnergyPlus init failed: {exc}"

	with tempfile.TemporaryDirectory(prefix="cuger_ep_run_") as run_tmp_dir:
		success, sql_path = run_single_simulation(
			idf_path=str(idf_path),
			epw_path=str(weather_path),
			output_dir=run_tmp_dir,
			modify_outputs=True,
			output_variables=None,
			diagnostics=None,
			verbose=False,
		)

		if not _valid_sql(success, sql_path):
			return False, tag, "simulation failed"

		try:
			reader = SQLReader(sql_path)
			out_csv = csv_dir / f"{tag}.csv"
			reader.export_external_loads_csv(str(out_csv))
			return True, tag, str(out_csv)
		except Exception as exc:
			return False, tag, f"SQL->CSV failed: {exc}"


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Batch run EnergyPlus for all IDF/EPW combinations and export CSV results"
	)
	parser.add_argument("--idf-dir", default=str(DEFAULT_IDF_DIR), help="Folder containing .idf files")
	parser.add_argument(
		"--epw-dir",
		default=str(DEFAULT_EPW_DIR),
		help="Folder containing weather files (.epw or .zip containing .epw)",
	)
	parser.add_argument("--csv-dir", default=str(DEFAULT_CSV_DIR), help="Output folder for .csv files")
	parser.add_argument(
		"--eplus-root",
		default=None,
		help="Optional EnergyPlus install root path; auto-detected if omitted",
	)
	parser.add_argument(
		"--workers",
		type=int,
		default=DEFAULT_WORKERS,
		help="Number of CPU workers for parallel runs (default: 8)",
	)
	parser.add_argument(
		"--idf-ratio",
		type=float,
		default=1.0,
		help="Random ratio (0,1] of IDF files to use",
	)
	parser.add_argument(
		"--weather-ratio",
		type=float,
		default=1.0,
		help="Random ratio (0,1] of weather files to use",
	)
	parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
	parser.add_argument("--quiet", action="store_true", help="Reduce log output")
	return parser


def main() -> int:
	args = build_parser().parse_args()

	idf_dir = Path(args.idf_dir).expanduser()
	epw_dir = Path(args.epw_dir).expanduser()
	csv_dir = Path(args.csv_dir).expanduser()
	try:
		csv_dir.mkdir(parents=True, exist_ok=True)
	except OSError as exc:
		print(f"Cannot access --csv-dir: {csv_dir}")
		print(f"OS error: {exc}")
		print("If this is a NAS path, authenticate first (for example, net use) or use an already-mounted drive path such as Z:/...")
		return 1

	idf_files = _collect_files(idf_dir, (".idf",))
	weather_files = _collect_files(epw_dir, (".epw", ".zip"))

	if not idf_files:
		print(f"No IDF files found in: {idf_dir}")
		return 1
	if not weather_files:
		print(f"No weather files (.epw/.zip) found in: {epw_dir}")
		return 1
	if not (0.0 < args.idf_ratio <= 1.0):
		print("--idf-ratio must be in (0, 1].")
		return 1
	if not (0.0 < args.weather_ratio <= 1.0):
		print("--weather-ratio must be in (0, 1].")
		return 1
	if args.workers < 1:
		print("--workers must be >= 1.")
		return 1

	rng = random.Random(args.seed)
	idf_files = _sample_files(idf_files, args.idf_ratio, rng)
	weather_files = _sample_files(weather_files, args.weather_ratio, rng)
	jobs = _build_jobs(idf_files, weather_files, rng)
	if not jobs:
		print("No simulation jobs generated after sampling and matching.")
		return 1

	try:
		exe_path, idd_path = locate_eplus(args.eplus_root)
		if not args.quiet:
			print(f"EnergyPlus executable: {exe_path}")
			print(f"EnergyPlus IDD: {idd_path}")
	except Exception as exc:
		print(f"Failed to locate EnergyPlus: {exc}")
		return 1

	total = len(jobs)
	done = 0
	ok = 0
	failed = 0
	logical_cpus = os.cpu_count() or 1
	workers = min(args.workers, total, logical_cpus)

	worker_eplus_root = str(Path(exe_path).parent)

	job_payloads = [
		(
			str(idf_path),
			str(weather_path),
			str(idf_dir),
			str(epw_dir),
			str(csv_dir),
			worker_eplus_root,
			args.quiet,
		)
		for idf_path, weather_path in jobs
	]

	try:
		pairs_csv = _write_job_pairs_csv(jobs, idf_dir, epw_dir, csv_dir)
	except OSError as exc:
		print(f"Failed to write job-pairs CSV: {exc}")
		return 1

	if not args.quiet:
		print(f"Total files: IDF={len(idf_files)}, Weather={len(weather_files)}, Jobs={total}")
		print(f"Job pairs CSV: {pairs_csv}")

	job_iter = iter(job_payloads)
	active_futures = set()

	with ProcessPoolExecutor(max_workers=workers) as executor:
		for _ in range(workers):
			try:
				active_futures.add(executor.submit(_run_job, *next(job_iter)))
			except StopIteration:
				break

		while active_futures:
			for future in as_completed(active_futures):
				active_futures.remove(future)
				done += 1
				try:
					success, _tag, _detail = future.result()
				except Exception:
					failed += 1
				else:
					if success:
						ok += 1
					else:
						failed += 1

				try:
					active_futures.add(executor.submit(_run_job, *next(job_iter)))
				except StopIteration:
					pass

				if not args.quiet:
					core_status = _format_core_status(len(active_futures), workers)
					print(
						f"\rProcessed files: {done}/{total} | Core status: {core_status}",
						end="",
						flush=True,
					)
				break

	if not args.quiet:
		print()

	print("\nBatch finished")
	print(f"  Success: {ok}")
	print(f"  Failed:  {failed}")
	return 0 if failed == 0 else 2


if __name__ == "__main__":
	raise SystemExit(main())
