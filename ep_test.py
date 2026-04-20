"""Batch-run EnergyPlus simulations and export SQL outputs to CSV.
"""

import argparse
import csv
import os
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from cuger.__simulate.simulate import locate_eplus, run_single_simulation  
from cuger.__simulate.dataset_new import (
	_build_graph_index,
	_find_graph_path,
	_normalize_building,
	_read_energy_npz,
	_read_weather_fallback,
	_save_building,
	_save_weather,
	build_packed_dataset_from_manifest,
)
from cuger.graphIO import json_to_graph
from cuger.__simulate.sqlread import SQLReader  

DEFAULT_IDF_DIR = Path(r"\\166.111.40.8\temp\lyh\SRT\All_DATA\idf")
DEFAULT_EPW_DIR = Path(r"\\166.111.40.8\temp\lyh\SRT\WEATHER")
DEFAULT_GRAPH_DIR = Path(r"\\166.111.40.8\temp\lyh\SRT\All_DATA\graph")
DEFAULT_PACK_DIR = Path(r"\\166.111.40.8\temp\lyh\SRT\All_DATA\pack_test")
DEFAULT_WORKERS = 8

_WORKER_EPLUS_READY = False
_WORKER_GRAPH_INDEX: dict[str, Path] | None = None


def _safe_stem(name: str) -> str:
	text = (name or "").strip().replace("/", "_").replace("\\", "_")
	for ch in [":", "*", "?", '"', "<", ">", "|"]:
		text = text.replace(ch, "_")
	return text or "unknown"


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


def _export_external_loads_npz(reader: SQLReader, out_npz: Path) -> None:
	if reader.zone_hourly_data is None:
		reader.read_zone_data()

	zone_ids = reader.get_zone_ids()
	zone_areas = reader.get_zone_areas(zone_ids)
	if not zone_ids:
		raise ValueError("No zones available to export.")

	data: dict[str, np.ndarray] = {}
	for zone_id in zone_ids:
		loads = reader.calculate_loads(zone_id)
		area = zone_areas.get(zone_id, None)
		if area is None or area <= 0:
			raise ValueError(f"Invalid area for zone {zone_id}")
		data[zone_id] = loads["external"] / area

	df = pd.DataFrame(data)
	values = df.to_numpy(dtype=np.float32, copy=True)
	columns = np.asarray([str(c) for c in df.columns], dtype=str)

	out_npz.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(out_npz, values=values, columns=columns)


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
    weather_ratio: float,
    rng: random.Random,
) -> list[tuple[Path, Path]]:
	k = max(1, int(round(len(weather_files) * weather_ratio)))
	k = min(k, len(weather_files))
	jobs: list[tuple[Path, Path]] = []
	for idf_path in idf_files:
		sampled_weather = rng.sample(weather_files, k)
		for weather_path in sampled_weather:
			jobs.append((idf_path, weather_path))
	return jobs


def _write_manifest_csv(
	jobs: list[tuple[Path, Path]],
	idf_dir: Path,
	epw_dir: Path,
	out_manifest_path: Path,
) -> Path:
	out_csv = out_manifest_path
	out_csv.parent.mkdir(parents=True, exist_ok=True)

	fieldnames = [
		"job_index",
		"idf_path",
		"weather_path",
		"idf_tag",
		"weather_tag",
		"job_tag",
		"sample_id",
		"energy_file",
		"weather_file",
		"building_file",
		"status",
		"result",
	]

	existing_job_tags: set[str] = set()
	next_index = 1
	if out_csv.exists():
		with out_csv.open("r", newline="", encoding="utf-8") as fp:
			reader = csv.DictReader(fp)
			for row in reader:
				job_tag = (row.get("job_tag") or "").strip()
				if job_tag:
					existing_job_tags.add(job_tag)
				idx_raw = (row.get("job_index") or "").strip()
				try:
					next_index = max(next_index, int(idx_raw) + 1)
				except (TypeError, ValueError):
					pass

	write_header = (not out_csv.exists()) or out_csv.stat().st_size == 0
	with out_csv.open("a", newline="", encoding="utf-8") as fp:
		writer = csv.writer(fp)
		if write_header:
			writer.writerow(fieldnames)
		for idf_path, weather_path in jobs:
			idf_tag = _build_tag(idf_path, idf_dir)
			weather_tag = _build_tag(weather_path, epw_dir)
			weather_name = _safe_stem(weather_path.stem) or _safe_stem(weather_tag)
			job_tag = f"{idf_tag}__{weather_tag}"
			if job_tag in existing_job_tags:
				continue
			writer.writerow([
				next_index,
				str(idf_path),
				str(weather_path),
				idf_tag,
				weather_tag,
				job_tag,
				job_tag,
				f"{job_tag}.npz",
				f"{weather_name}.npz",
				f"{idf_tag}.npz",
				"pending",
				0,
			])
			next_index += 1
			existing_job_tags.add(job_tag)
	return out_csv


def _read_manifest_csv(pairs_csv_path: Path) -> list[dict[str, str]]:
	jobs: list[dict[str, str]] = []
	with pairs_csv_path.open("r", newline="", encoding="utf-8") as fp:
		reader = csv.DictReader(fp)
		for row in reader:
			if not row.get("idf_path") or not row.get("weather_path") or not row.get("job_tag"):
				continue
			jobs.append(row)
	return jobs


def _read_manifest(path: Path) -> list[dict[str, str]]:
	if path.suffix.lower() == ".csv":
		return _read_manifest_csv(path)
	else:
		raise ValueError(f"Unsupported job file format: {path}. Only CSV format is supported.")


def _update_manifest_status(manifest_path: Path, job_tag: str, status: str) -> None:
	"""Update the status of a specific job in manifest CSV format."""
	_update_manifest_status_csv(manifest_path, job_tag, status)


def _update_manifest_status_csv(manifest_path: Path, job_tag: str, status: str) -> None:
	"""Update the status of a specific job in CSV format."""
	import tempfile
	import shutil
	
	# Determine result value based on status
	result = 1 if status == "completed" else (-1 if status == "failed" else 0)
	
	# Read all rows
	rows = []
	with manifest_path.open("r", newline="", encoding="utf-8") as fp:
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
		shutil.move(tmp_fp.name, manifest_path)


def _filter_completed_jobs(job_records: list[dict[str, str]]) -> tuple[list[tuple[Path, Path]], int]:
	pending: list[tuple[Path, Path]] = []
	completed = 0
	for row in job_records:
		job_tag = row.get("job_tag", "")
		status = (row.get("status", "pending") or "pending").strip().lower()
		result = (row.get("result", "0") or "0").strip()
		idf_path = row.get("idf_path")
		weather_path = row.get("weather_path")
		if not idf_path or not weather_path:
			continue

		# Respect explicit job status first.
		# Pending jobs should always be executed when resuming.
		if status == "pending":
			pending.append((Path(idf_path), Path(weather_path)))
			continue
		if status in ("completed", "failed"):
			completed += 1
			continue

		# Fallback for legacy rows where status was not reliably maintained.
		try:
			result_num = int(result)
			if result_num == -1:
				completed += 1
				continue
			if result_num == 1:
				completed += 1
				continue
		except (ValueError, TypeError):
			pass  # Treat invalid result values as pending.

		pending.append((Path(idf_path), Path(weather_path)))
	return pending, completed


def _ensure_worker_eplus(eplus_root_str: str | None) -> None:
	global _WORKER_EPLUS_READY
	if _WORKER_EPLUS_READY:
		return
	locate_eplus(eplus_root_str or None)
	_WORKER_EPLUS_READY = True


def _get_worker_graph_index(graph_dir: Path) -> dict[str, Path]:
	global _WORKER_GRAPH_INDEX
	if _WORKER_GRAPH_INDEX is None:
		_WORKER_GRAPH_INDEX = _build_graph_index(graph_dir)
	return _WORKER_GRAPH_INDEX


def _export_sidecar_npz(
	idf_path: Path,
	idf_dir: Path,
	graph_dir: Path,
	weather_path: Path,
	out_energy_npz: Path,
	out_weather_npz: Path,
	out_building_npz: Path,
) -> None:
	graph_index = _get_worker_graph_index(graph_dir)
	idf_tag = _build_tag(idf_path, idf_dir)
	graph_path = _find_graph_path(graph_index, idf_stem=idf_path.stem, idf_tag=idf_tag)
	if graph_path is None:
		raise FileNotFoundError(f"graph not found for idf={idf_path.name}")

	weather_df = _read_weather_fallback(weather_path)
	energy_df = _read_energy_npz(out_energy_npz)
	building_graph = json_to_graph(str(graph_path))
	norm_building = _normalize_building(building_graph, energy_columns=list(energy_df.columns))

	out_weather_npz.parent.mkdir(parents=True, exist_ok=True)
	out_building_npz.parent.mkdir(parents=True, exist_ok=True)
	if not out_weather_npz.exists():
		_save_weather(out_weather_npz, weather_df)
	if not out_building_npz.exists():
		_save_building(out_building_npz, norm_building)


def _run_job(
	idf_path_str: str,
	weather_path_str: str,
	idf_dir_str: str,
	epw_dir_str: str,
	energy_dir_str: str,
	weather_npz_dir_str: str,
	building_npz_dir_str: str,
	graph_dir_str: str,
	eplus_root_str: str,
	quiet: bool,
) -> tuple[bool, str, str]:
	idf_path = Path(idf_path_str)
	weather_path = Path(weather_path_str)
	idf_dir = Path(idf_dir_str)
	epw_dir = Path(epw_dir_str)
	energy_dir = Path(energy_dir_str)
	weather_npz_dir = Path(weather_npz_dir_str)
	building_npz_dir = Path(building_npz_dir_str)
	graph_dir = Path(graph_dir_str)

	idf_tag = _build_tag(idf_path, idf_dir)
	weather_tag = _build_tag(weather_path, epw_dir)
	tag = f"{idf_tag}__{weather_tag}"
	weather_name = _safe_stem(weather_path.stem) or _safe_stem(weather_tag)
	out_npz = energy_dir / f"{tag}.npz"
	out_weather_npz = weather_npz_dir / f"{weather_name}.npz"
	out_building_npz = building_npz_dir / f"{idf_tag}.npz"

	if out_npz.exists() and out_weather_npz.exists() and out_building_npz.exists():
		return True, tag, "skip existing npz"

	if out_npz.exists():
		try:
			_export_sidecar_npz(
				idf_path=idf_path,
				idf_dir=idf_dir,
				graph_dir=graph_dir,
				weather_path=weather_path,
				out_energy_npz=out_npz,
				out_weather_npz=out_weather_npz,
				out_building_npz=out_building_npz,
			)
			return True, tag, "skip simulation, completed missing sidecar npz"
		except Exception as exc:
			return False, tag, f"existing energy sidecar export failed: {exc}"

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
			if not out_npz.exists():
				_export_external_loads_npz(reader, out_npz)
			_export_sidecar_npz(
				idf_path=idf_path,
				idf_dir=idf_dir,
				graph_dir=graph_dir,
				weather_path=weather_path,
				out_energy_npz=out_npz,
				out_weather_npz=out_weather_npz,
				out_building_npz=out_building_npz,
			)
			return True, tag, str(out_npz)
		except Exception as exc:
			return False, tag, f"NPZ export failed: {exc}"
		


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Batch run EnergyPlus for all IDF/EPW combinations and export energy NPZ results"
	)
	parser.add_argument("--idf-dir", default=str(DEFAULT_IDF_DIR), help="Folder containing .idf files")
	parser.add_argument(
		"--epw-dir",
		default=str(DEFAULT_EPW_DIR),
		help="Folder containing weather files (.epw or .zip containing .epw)",
	)
	parser.add_argument("--energy-dir", default=str(DEFAULT_PACK_DIR / "energy"), help="Folder for per-job energy .npz files")
	parser.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR), help="Folder containing graph .json files")
	parser.add_argument("--pack-dir", default=None, help="Folder containing manifest file (default: All_DATA/pack)")
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
	parser.add_argument("--resume", action="store_true", help="Resume from existing jobs manifest file")
	parser.add_argument("--manifest-file", default="jobs_manifest.csv", help="Jobs manifest CSV filename under --pack-dir, default: jobs_manifest.csv")
	parser.add_argument("--job-file", default=None, help="Deprecated alias for --manifest-file (jobs manifest)")
	parser.add_argument("--log-interval", type=int, default=100, help="Print progress every N completed jobs (default:100)")
	return parser


def main() -> int:
	args = build_parser().parse_args()

	idf_dir = Path(args.idf_dir).expanduser()
	epw_dir = Path(args.epw_dir).expanduser()
	energy_dir = Path(args.energy_dir).expanduser()
	graph_dir = Path(args.graph_dir).expanduser()
	pack_dir = Path(args.pack_dir).expanduser() if args.pack_dir else DEFAULT_PACK_DIR
	weather_npz_dir = energy_dir.parent / "weather"
	building_npz_dir = energy_dir.parent / "building"
	try:
		energy_dir.mkdir(parents=True, exist_ok=True)
		weather_npz_dir.mkdir(parents=True, exist_ok=True)
		building_npz_dir.mkdir(parents=True, exist_ok=True)
	except OSError as exc:
		print(f"Cannot access --energy-dir: {energy_dir}")
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

	manifest_filename = args.manifest_file if args.manifest_file else (args.job_file or "jobs_manifest.csv")
	if args.job_file and not args.quiet:
		print("Warning: --job-file is deprecated, use --manifest-file instead.")
	jobs_manifest_path = Path(manifest_filename)
	if not jobs_manifest_path.is_absolute():
		jobs_manifest_path = pack_dir / jobs_manifest_path

	jobs: list[tuple[Path, Path]] = []
	existing_done = 0
	skip_simulation = False

	if jobs_manifest_path.exists() and args.resume:
		if not args.quiet:
			print(f"Resuming from existing jobs manifest: {jobs_manifest_path}")
		job_records = _read_manifest(jobs_manifest_path)
		jobs, existing_done = _filter_completed_jobs(job_records)
		if not args.quiet:
			print(f"Existing {len(job_records)} jobs, {existing_done} already done, {len(jobs)} pending")
		if not jobs:
			skip_simulation = True
			if not args.quiet:
				print("All jobs already completed. Skip simulation and continue to dataset step.")
	else:
		if jobs_manifest_path.exists() and not args.resume and not args.quiet:
			print(f"Appending new rows into existing jobs manifest: {jobs_manifest_path}")
		jobs = _build_jobs(idf_files, weather_files, args.weather_ratio, rng)
		if not jobs:
			print("No simulation jobs generated after sampling and matching.")
			return 1
		try:
			_write_manifest_csv(jobs, idf_dir, epw_dir, jobs_manifest_path)
		except OSError as exc:
			print(f"Failed to write jobs manifest CSV: {exc}")
			return 1

	total_pending = len(jobs)
	done = 0
	ok = 0
	failed = 0

	if not skip_simulation:
		try:
			exe_path, idd_path = locate_eplus(args.eplus_root)
			if not args.quiet:
				print(f"EnergyPlus executable: {exe_path}")
				print(f"EnergyPlus IDD: {idd_path}")
		except Exception as exc:
			print(f"Failed to locate EnergyPlus: {exc}")
			return 1

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
				str(energy_dir),
				str(weather_npz_dir),
				str(building_npz_dir),
				str(graph_dir),
				worker_eplus_root,
				args.quiet,
			)
			for idf_path, weather_path in jobs
		]

		if not args.quiet:
			print(f"Total files: IDF={len(idf_files)}, Weather={len(weather_files)}, Pending jobs={total_pending}, Completed before run={existing_done}")
			print(f"Jobs manifest file: {jobs_manifest_path}")

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
							try:
								_update_manifest_status(jobs_manifest_path, _tag, "completed")
							except Exception:
								pass
						else:
							failed += 1
							try:
								_update_manifest_status(jobs_manifest_path, _tag, "failed")
							except Exception:
								pass

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

	pack_stats = build_packed_dataset_from_manifest(
		manifest_path=jobs_manifest_path,
		energy_dir=energy_dir,
		graph_dir=graph_dir,
		pack_dir=pack_dir,
		epw_dir=epw_dir,
		skip_existing=False,
		quiet=args.quiet,
	)
	print("Pack step finished")
	print(f"  Pack dir:              {pack_dir}")
	print(f"  Output sample manifest: {pack_dir / 'manifest.csv'}")
	print(f"  Eligible jobs:         {pack_stats['eligible']}")
	print(f"  Newly built samples:   {pack_stats['built']}")
	print(f"  Skipped existing:      {pack_stats['skipped_existing']}")
	print(f"  Skipped missing input: {pack_stats['skipped_missing_input']}")
	print(f"  Failed in pack step:   {pack_stats['failed']}")
	print(f"  Unique weather files:  {pack_stats['weather_unique']}")
	print(f"  Unique building files: {pack_stats['building_unique']}")

	if failed > 0:
		return 2
	if pack_stats["failed"] > 0:
		return 3
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
