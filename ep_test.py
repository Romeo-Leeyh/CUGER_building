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


DEFAULT_IDF_DIR = Path(r"Z:\lyh\SRT\All_DATA_SRT\idf")
DEFAULT_EPW_DIR = Path(r"Z:\lyh\SRT\WEATHER")
DEFAULT_CSV_DIR = Path(r"Z:\lyh\SRT\All_DATA_SRT\data")
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
		writer.writerow([
			"job_index",
			"idf_path",
			"weather_path",
			"idf_tag",
			"weather_tag",
			"job_tag",
			"status",
			"result",
		])
		for idx, (idf_path, weather_path) in enumerate(jobs, start=1):
			idf_tag = _build_tag(idf_path, idf_dir)
			weather_tag = _build_tag(weather_path, epw_dir)
			job_tag = f"{idf_tag}__{weather_tag}"
			writer.writerow([idx, str(idf_path), str(weather_path), idf_tag, weather_tag, job_tag, "pending", 0])
	return out_csv


def _read_job_pairs_csv(pairs_csv_path: Path) -> list[dict[str, str]]:
	jobs: list[dict[str, str]] = []
	with pairs_csv_path.open("r", newline="", encoding="utf-8") as fp:
		reader = csv.DictReader(fp)
		for row in reader:
			if not row.get("idf_path") or not row.get("weather_path") or not row.get("job_tag"):
				continue
			jobs.append(row)
	return jobs


def _read_job_pairs(path: Path) -> list[dict[str, str]]:
	if path.suffix.lower() == ".csv":
		return _read_job_pairs_csv(path)
	else:
		raise ValueError(f"Unsupported job file format: {path}. Only CSV format is supported.")


def _update_job_status(job_pairs_path: Path, job_tag: str, status: str) -> None:
	"""Update the status of a specific job in the job pairs file."""
	_update_job_status_csv(job_pairs_path, job_tag, status)


def _update_job_status_csv(job_pairs_path: Path, job_tag: str, status: str) -> None:
	"""Update the status of a specific job in CSV format."""
	import tempfile
	import shutil
	
	# Determine result value based on status
	result = 1 if status == "completed" else (-1 if status == "failed" else 0)
	
	# Read all rows
	rows = []
	with job_pairs_path.open("r", newline="", encoding="utf-8") as fp:
		reader = csv.DictReader(fp)
		fieldnames = reader.fieldnames
		for row in reader:
			if row.get("job_tag") == job_tag:
				row["status"] = status
				row["result"] = str(result)
			rows.append(row)
	
	# Write back with updated status
	with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", delete=False, suffix=".csv") as tmp_fp:
		writer = csv.DictWriter(tmp_fp, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)
		tmp_fp.flush()
		shutil.move(tmp_fp.name, job_pairs_path)


def _filter_completed_jobs(job_records: list[dict[str, str]], csv_dir: Path) -> tuple[list[tuple[Path, Path]], int]:
	pending: list[tuple[Path, Path]] = []
	completed = 0
	for row in job_records:
		job_tag = row.get("job_tag", "")
		status = row.get("status", "pending")
		result = row.get("result", "0")
		idf_path = row.get("idf_path")
		weather_path = row.get("weather_path")
		if not idf_path or not weather_path:
			continue
		
		# Check if output file exists (primary check)
		out_csv = csv_dir / f"{job_tag}.csv"
		if out_csv.exists():
			completed += 1
			continue
		
		# Also consider completed/failed status as done (don't retry)
		if status in ("completed", "failed"):
			completed += 1
			continue
		
		# Also consider result -1 (failed) as done (don't retry failed jobs)
		try:
			if int(result) == -1:
				completed += 1
				continue
		except (ValueError, TypeError):
			pass  # Ignore invalid result values
			
		pending.append((Path(idf_path), Path(weather_path)))
	return pending, completed


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
	parser.add_argument("--resume", action="store_true", help="Resume from existing job_pairs file in --csv-dir")
	parser.add_argument("--job-file", default="job_pairs.csv", help="Job pairs metadata file (CSV format), default: job_pairs.csv")
	parser.add_argument("--log-interval", type=int, default=100, help="Print progress every N completed jobs (default:100)")
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

	job_pairs_path = Path(args.job_file)
	if not job_pairs_path.is_absolute():
		job_pairs_path = csv_dir / job_pairs_path

	jobs: list[tuple[Path, Path]] = []
	existing_done = 0

	if job_pairs_path.exists() and args.resume:
		if not args.quiet:
			print(f"Resuming from existing job pairs: {job_pairs_path}")
		job_records = _read_job_pairs(job_pairs_path)
		jobs, existing_done = _filter_completed_jobs(job_records, csv_dir)
		if not args.quiet:
			print(f"Existing {len(job_records)} jobs, {existing_done} already done, {len(jobs)} pending")
		if not jobs:
			print("All jobs already completed. Exiting.")
			return 0
	else:
		if job_pairs_path.exists() and not args.resume and not args.quiet:
			print(f"Overwriting {job_pairs_path} in {csv_dir}")
		jobs = _build_jobs(idf_files, weather_files, rng)
		if not jobs:
			print("No simulation jobs generated after sampling and matching.")
			return 1
		try:
			_write_job_pairs_csv(jobs, idf_dir, epw_dir, csv_dir)
		except OSError as exc:
			print(f"Failed to write job-pairs CSV: {exc}")
			return 1

	try:
		exe_path, idd_path = locate_eplus(args.eplus_root)
		if not args.quiet:
			print(f"EnergyPlus executable: {exe_path}")
			print(f"EnergyPlus IDD: {idd_path}")
	except Exception as exc:
		print(f"Failed to locate EnergyPlus: {exc}")
		return 1

	total_pending = len(jobs)
	done = 0
	ok = 0
	failed = 0
	logical_cpus = os.cpu_count() or 1
	workers = min(args.workers, total_pending, logical_cpus)
	overall_total = existing_done + total_pending

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

	if not args.quiet:
		print(f"Total files: IDF={len(idf_files)}, Weather={len(weather_files)}, Pending jobs={total_pending}, Completed before run={existing_done}")
		print(f"Job pairs file: {job_pairs_path}")

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
				except Exception as exc:
					failed += 1
					# Update status to failed
					try:
						_update_job_status(job_pairs_path, _tag, "failed")
					except Exception:
						pass  # Ignore update errors
				else:
					if success:
						ok += 1
						# Update status to completed
						try:
							_update_job_status(job_pairs_path, _tag, "completed")
						except Exception:
							pass  # Ignore update errors
					else:
						failed += 1
						# Update status to failed
						try:
							_update_job_status(job_pairs_path, _tag, "failed")
						except Exception:
							pass  # Ignore update errors

				try:
					active_futures.add(executor.submit(_run_job, *next(job_iter)))
				except StopIteration:
					pass

				if not args.quiet and (done % args.log_interval == 0 or done == total_pending):
					core_status = _format_core_status(len(active_futures), workers)
					processed = existing_done + done
					print(
						f"\rProcessed files: {processed}/{overall_total} | Core status: {core_status}",
						end="",
						flush=True,
					)
				break

	if not args.quiet:
		print()

	print("\nBatch finished")
	print(f"  Completed before run: {existing_done}")
	print(f"  Success in this run: {ok}")
	print(f"  Failed in this run:  {failed}")
	print(f"  Total attempted in this run: {ok + failed}")
	return 0 if failed == 0 else 2


if __name__ == "__main__":
	raise SystemExit(main())
