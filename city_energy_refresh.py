"""Recompute city-related EnergyPlus cases and overwrite selected energy NPZ files.

Workflow:
1. Read one manifest CSV file.
2. Filter rows related to one city keyword.
3. Resolve corresponding EPW, IDF, graph, and target energy NPZ paths.
4. Run EnergyPlus simulation per case.
5. Export external loads, align energy columns by graph valid spaces, and
   overwrite target NPZ in PACK/energy.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cuger.__simulate.dataset_new import _build_graph_index, _find_graph_path, _normalize_building
from cuger.__simulate.simulate import locate_eplus, run_single_simulation
from cuger.__simulate.sqlread import SQLReader
from cuger.graphIO import json_to_graph


DEFAULT_MANIFEST_FILE = Path(r"\\166.111.40.8\temp\lyh\SRT\PACK\manifest.csv")
DEFAULT_WEATHER_DIR = Path(r"\\166.111.40.8\temp\lyh\SRT\WEATHER")
DEFAULT_IDF_DIR = Path(r"\\166.111.40.8\temp\lyh\SRT\PACK\idf")
DEFAULT_GRAPH_DIR = Path(r"\\166.111.40.8\temp\lyh\SRT\PACK\graph")
DEFAULT_ENERGY_DIR = Path(r"\\166.111.40.8\temp\lyh\SRT\PACK\energy")
# Edit this list directly when you want fixed in-script city filters.
DEFAULT_CITIES = [
	"chicago",
    "amsterdam",
    "athens",
    "cairo",
    "casablanca",
    "delhi",
    "lagos",
    "lima",
    "paris",
    "prague",
    "rome",
    "santiago",
    "warsaw",
    "wellington",
]

_WORKER_EPLUS_READY = False


@dataclass
class CaseRecord:
	manifest_path: Path
	row_index: int
	case_id: str
	idf_path: Path
	weather_path: Path
	graph_path: Path
	energy_npz_path: Path


def _ensure_worker_eplus(eplus_root: str | None) -> None:
	global _WORKER_EPLUS_READY
	if _WORKER_EPLUS_READY:
		return
	locate_eplus(eplus_root or None)
	_WORKER_EPLUS_READY = True


def _normalize_text(text: str) -> str:
	return "".join(ch for ch in (text or "").casefold() if ch.isalnum())


def _city_related(row: dict[str, str], city: str) -> bool:
	city_norm = _normalize_text(city)
	if not city_norm:
		return False

	for value in row.values():
		if city_norm in _normalize_text(str(value)):
			return True
	return False


def _city_list_related(row: dict[str, str], cities: list[str]) -> bool:
	for city in cities:
		if _city_related(row, city):
			return True
	return False


def _collect_files(folder: Path, suffixes: tuple[str, ...]) -> list[Path]:
	if not folder.is_dir():
		return []
	return sorted(
		p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in suffixes
	)


def _file_safe_token(text: str) -> str:
	t = (text or "").strip().replace("/", "_").replace("\\", "_")
	for ch in [":", "*", "?", '"', "<", ">", "|"]:
		t = t.replace(ch, "_")
	return t


def _build_lookup(files: list[Path]) -> dict[str, list[Path]]:
	index: dict[str, list[Path]] = {}
	for p in files:
		index.setdefault(p.stem.casefold(), []).append(p)
		index.setdefault(p.name.casefold(), []).append(p)
	return index


def _first_existing(paths: list[Path]) -> Path | None:
	for p in paths:
		if p and p.exists():
			return p
	return None


def _extract_weather_stem_candidates(row: dict[str, str]) -> list[str]:
	keys = ["weather_path", "weather_file", "weather_id", "weather_tag", "sample_id", "job_tag"]
	cands: list[str] = []
	for k in keys:
		raw = (row.get(k) or "").strip()
		if not raw:
			continue
		p = Path(raw)
		if p.stem:
			cands.append(_file_safe_token(p.stem))
		if p.name:
			cands.append(_file_safe_token(p.name))
	return cands


def _extract_idf_stem_candidates(row: dict[str, str]) -> list[str]:
	keys = ["idf_path", "building_file", "building_id", "idf_tag", "sample_id", "job_tag"]
	cands: list[str] = []
	for k in keys:
		raw = (row.get(k) or "").strip()
		if not raw:
			continue
		p = Path(raw)
		if p.stem:
			cands.append(_file_safe_token(p.stem))
		if p.name:
			cands.append(_file_safe_token(p.name))
	return cands


def _resolve_weather_path(row: dict[str, str], weather_dir: Path, weather_lookup: dict[str, list[Path]]) -> Path | None:
	raw_weather = (row.get("weather_path") or "").strip()
	if raw_weather:
		p = Path(raw_weather)
		direct = _first_existing([p, weather_dir / p.name])
		if direct is not None:
			return direct

	for cand in _extract_weather_stem_candidates(row):
		matches = weather_lookup.get(cand.casefold(), [])
		if matches:
			return sorted(matches)[0]

	return None


def _resolve_idf_path(row: dict[str, str], idf_dir: Path, idf_lookup: dict[str, list[Path]]) -> Path | None:
	raw_idf = (row.get("idf_path") or "").strip()
	if raw_idf:
		p = Path(raw_idf)
		direct = _first_existing([p, idf_dir / p.name])
		if direct is not None:
			return direct

	for cand in _extract_idf_stem_candidates(row):
		matches = idf_lookup.get(cand.casefold(), [])
		if matches:
			return sorted(matches)[0]

	return None


def _resolve_energy_path(row: dict[str, str], energy_dir: Path) -> Path | None:
	for key in ["energy_file", "job_tag", "sample_id"]:
		raw = (row.get(key) or "").strip()
		if not raw:
			continue
		if key == "energy_file":
			name = Path(raw).name
		else:
			name = f"{raw}.npz"
		candidate = energy_dir / name
		if candidate.exists():
			return candidate
	return None


def _read_city_rows(manifest_csv: Path, cities: list[str]) -> list[tuple[int, dict[str, str]]]:
	rows: list[tuple[int, dict[str, str]]] = []
	with manifest_csv.open("r", newline="", encoding="utf-8") as fp:
		reader = csv.DictReader(fp)
		for idx, row in enumerate(reader, start=2):
			if _city_list_related(row, cities):
				rows.append((idx, row))
	return rows


def _export_external_loads_df(reader: SQLReader) -> pd.DataFrame:
	if reader.zone_hourly_data is None:
		reader.read_zone_data()

	zone_ids = reader.get_zone_ids()
	zone_areas = reader.get_zone_areas(zone_ids)
	if not zone_ids:
		raise ValueError("No zones available in SQL output")

	data: dict[str, np.ndarray] = {}
	for zone_id in zone_ids:
		loads = reader.calculate_loads(zone_id)
		area = zone_areas.get(zone_id)
		if area is None or area <= 0:
			raise ValueError(f"Invalid area for zone: {zone_id}")
		data[str(zone_id)] = loads["external"] / area

	return pd.DataFrame(data)


def _align_energy_by_graph(energy_df: pd.DataFrame, graph_path: Path) -> pd.DataFrame:
	building_graph = json_to_graph(str(graph_path))
	norm_building = _normalize_building(building_graph, energy_columns=list(energy_df.columns))

	valid_spaces = [str(v) for v in norm_building["valid_energy_spaces"].tolist()]
	valid_spaces = [s for s in valid_spaces if s in energy_df.columns]
	if not valid_spaces:
		raise ValueError("No overlap between graph valid_energy_spaces and energy columns")

	return energy_df[valid_spaces]


def _save_energy_npz(path: Path, energy_df: pd.DataFrame) -> None:
	values = energy_df.to_numpy(dtype=np.float32, copy=True)
	columns = np.asarray([str(c) for c in energy_df.columns], dtype=str)
	path.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(path, values=values, columns=columns)


def _build_case_id(row: dict[str, str], fallback: str) -> str:
	for key in ["sample_id", "job_tag", "energy_file"]:
		v = (row.get(key) or "").strip()
		if v:
			return v
	return fallback


def collect_cases(
	cities: list[str],
	manifest_csv: Path,
	weather_dir: Path,
	idf_dir: Path,
	graph_dir: Path,
	energy_dir: Path,
	allow_missing_energy: bool,
) -> tuple[list[CaseRecord], list[str]]:
	errors: list[str] = []
	weather_files = _collect_files(weather_dir, (".epw", ".zip"))
	idf_files = _collect_files(idf_dir, (".idf",))

	weather_lookup = _build_lookup(weather_files)
	idf_lookup = _build_lookup(idf_files)
	graph_index = _build_graph_index(graph_dir)

	cases: list[CaseRecord] = []

	try:
		rows = _read_city_rows(manifest_csv, cities)
	except Exception as exc:
		errors.append(f"{manifest_csv}: failed to read ({exc})")
		return [], errors

	for row_index, row in rows:
		case_id = _build_case_id(row, fallback=f"{manifest_csv.stem}:{row_index}")

		idf_path = _resolve_idf_path(row, idf_dir, idf_lookup)
		if idf_path is None:
			errors.append(f"{manifest_csv}:{row_index}: idf not resolved")
			continue

		weather_path = _resolve_weather_path(row, weather_dir, weather_lookup)
		if weather_path is None:
			errors.append(f"{manifest_csv}:{row_index}: weather not resolved")
			continue

		idf_tag = (row.get("idf_tag") or "").strip()
		graph_path = _find_graph_path(graph_index, idf_stem=idf_path.stem, idf_tag=idf_tag)
		if graph_path is None:
			errors.append(f"{manifest_csv}:{row_index}: graph not found for {idf_path.name}")
			continue

		energy_path = _resolve_energy_path(row, energy_dir)
		if energy_path is None:
			if not allow_missing_energy:
				errors.append(f"{manifest_csv}:{row_index}: target energy npz not found")
				continue
			energy_name = f"{_file_safe_token(case_id)}.npz"
			energy_path = energy_dir / energy_name

		cases.append(
			CaseRecord(
				manifest_path=manifest_csv,
				row_index=row_index,
				case_id=case_id,
				idf_path=idf_path,
				weather_path=weather_path,
				graph_path=graph_path,
				energy_npz_path=energy_path,
			)
		)

	uniq: dict[tuple[str, str], CaseRecord] = {}
	for c in cases:
		uniq[(str(c.idf_path), str(c.weather_path))] = c
	return list(uniq.values()), errors


def run_case(case: CaseRecord, eplus_root: str | None, quiet: bool) -> tuple[bool, str]:
	try:
		_ensure_worker_eplus(eplus_root)
	except Exception as exc:
		return False, f"EnergyPlus init failed: {exc}"

	with tempfile.TemporaryDirectory(prefix="city_ep_refresh_") as tmp_dir:
		success, sql_path = run_single_simulation(
			idf_path=str(case.idf_path),
			epw_path=str(case.weather_path),
			output_dir=tmp_dir,
			modify_outputs=True,
			output_variables=None,
			diagnostics=None,
			verbose=False,
		)
		if not success or not sql_path or not Path(sql_path).is_file():
			return False, "simulation failed or sql missing"

		reader = SQLReader(sql_path)
		energy_df = _export_external_loads_df(reader)
		aligned_df = _align_energy_by_graph(energy_df, case.graph_path)
		_save_energy_npz(case.energy_npz_path, aligned_df)

	return True, "ok"


def _format_core_status(active_workers: int, workers: int) -> str:
	labels: list[str] = []
	for idx in range(workers):
		state = "RUN" if idx < active_workers else "IDLE"
		labels.append(f"C{idx + 1}:{state}")
	return " ".join(labels)


def _run_case_payload(
	manifest_path_str: str,
	row_index: int,
	case_id: str,
	idf_path_str: str,
	weather_path_str: str,
	graph_path_str: str,
	energy_npz_path_str: str,
	eplus_root: str | None,
	quiet: bool,
) -> tuple[bool, str, str, str, int]:
	case = CaseRecord(
		manifest_path=Path(manifest_path_str),
		row_index=row_index,
		case_id=case_id,
		idf_path=Path(idf_path_str),
		weather_path=Path(weather_path_str),
		graph_path=Path(graph_path_str),
		energy_npz_path=Path(energy_npz_path_str),
	)
	success, detail = run_case(case=case, eplus_root=eplus_root, quiet=quiet)
	return success, detail, case.case_id, str(case.manifest_path), case.row_index


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Recompute city-related EnergyPlus cases and overwrite PACK/energy NPZ files"
	)
	parser.add_argument(
		"cities",
		nargs="*",
		default=None,
		help="Optional city keywords. Provide multiple values like: beijing shanghai. If omitted, DEFAULT_CITIES in script is used.",
	)
	parser.add_argument("--manifest-file", default=str(DEFAULT_MANIFEST_FILE), help="Manifest CSV file path")
	parser.add_argument("--weather-dir", default=str(DEFAULT_WEATHER_DIR), help="Folder containing .epw/.zip weather files")
	parser.add_argument("--idf-dir", default=str(DEFAULT_IDF_DIR), help="Folder containing .idf files")
	parser.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR), help="Folder containing graph .json files")
	parser.add_argument("--energy-dir", default=str(DEFAULT_ENERGY_DIR), help="Folder containing target energy .npz files")
	parser.add_argument("--eplus-root", default=None, help="Optional EnergyPlus installation root")
	parser.add_argument(
		"--workers",
		type=int,
		default=1,
		help="Number of worker processes for parallel simulation (default: 1)",
	)
	parser.add_argument(
		"--start-index",
		type=int,
		default=1,
		help="1-based start index in matched cases (default: 1)",
	)
	parser.add_argument(
		"--end-index",
		type=int,
		default=0,
		help="1-based end index in matched cases, inclusive (0 means until end)",
	)
	parser.add_argument("--limit", type=int, default=0, help="Optional max number of matched cases to run (0 means no limit)")
	parser.add_argument(
		"--allow-missing-energy",
		action="store_true",
		help="If set, create new npz when target energy file is not found",
	)
	parser.add_argument("--quiet", action="store_true", help="Reduce output logs")
	return parser


def main() -> int:
	args = build_parser().parse_args()

	cities = [c for c in (args.cities or DEFAULT_CITIES) if str(c).strip()]
	if not cities:
		print("No city keywords provided. Set DEFAULT_CITIES in script or pass cities in CLI.")
		return 1

	manifest_file = Path(args.manifest_file).expanduser()
	weather_dir = Path(args.weather_dir).expanduser()
	idf_dir = Path(args.idf_dir).expanduser()
	graph_dir = Path(args.graph_dir).expanduser()
	energy_dir = Path(args.energy_dir).expanduser()

	if not manifest_file.is_file() or manifest_file.suffix.lower() != ".csv":
		print(f"Manifest CSV file not found or not csv: {manifest_file}")
		return 1
	if not weather_dir.is_dir():
		print(f"Weather folder not found: {weather_dir}")
		return 1
	if not idf_dir.is_dir():
		print(f"IDF folder not found: {idf_dir}")
		return 1
	if not graph_dir.is_dir():
		print(f"Graph folder not found: {graph_dir}")
		return 1
	if not energy_dir.is_dir():
		print(f"Energy folder not found: {energy_dir}")
		return 1
	if args.workers < 1:
		print("--workers must be >= 1")
		return 1
	if args.start_index < 1:
		print("--start-index must be >= 1")
		return 1
	if args.end_index < 0:
		print("--end-index must be >= 0")
		return 1
	if args.end_index > 0 and args.end_index < args.start_index:
		print("--end-index must be >= --start-index when provided")
		return 1

	has_existing_energy_npz = any(energy_dir.rglob("*.npz"))
	effective_allow_missing_energy = args.allow_missing_energy or (not has_existing_energy_npz)
	if (not args.quiet) and (not has_existing_energy_npz) and (not args.allow_missing_energy):
		print("Energy dir has no existing npz; auto-enable missing-energy creation for bootstrap run.")

	try:
		exe_path, idd_path = locate_eplus(args.eplus_root)
		if not args.quiet:
			print(f"EnergyPlus executable: {exe_path}")
			print(f"EnergyPlus IDD: {idd_path}")
	except Exception as exc:
		print(f"Failed to locate EnergyPlus: {exc}")
		return 1

	cases, resolve_errors = collect_cases(
		cities=cities,
		manifest_csv=manifest_file,
		weather_dir=weather_dir,
		idf_dir=idf_dir,
		graph_dir=graph_dir,
		energy_dir=energy_dir,
		allow_missing_energy=effective_allow_missing_energy,
	)
	matched_total = len(cases)

	start0 = args.start_index - 1
	if args.end_index > 0:
		cases = cases[start0 : args.end_index]
	else:
		cases = cases[start0:]

	if args.limit and args.limit > 0:
		cases = cases[: args.limit]

	if not args.quiet:
		print(f"City keywords: {', '.join(cities)}")
		print(f"Matched runnable cases (before range): {matched_total}")
		if args.end_index > 0:
			print(f"Selected range: {args.start_index}..{args.end_index}")
		else:
			print(f"Selected range: {args.start_index}..end")
		print(f"Runnable cases after range/limit: {len(cases)}")
		print(f"Workers: {args.workers}")
		if resolve_errors:
			print(f"Skipped rows during resolution: {len(resolve_errors)}")

	if not cases:
		print("No runnable cases found.")
		if resolve_errors and not args.quiet:
			for msg in resolve_errors[:20]:
				print(f"  - {msg}")
			if len(resolve_errors) > 20:
				print(f"  ... and {len(resolve_errors) - 20} more")
		return 2

	ok = 0
	failed = 0
	total = len(cases)
	logical_cpus = os.cpu_count() or 1
	workers = min(args.workers, logical_cpus, total)

	eplus_worker_root = str(Path(exe_path).parent)

	if workers <= 1:
		for idx, case in enumerate(cases, start=1):
			if not args.quiet:
				print(f"[{idx}/{total}] Running case: {case.case_id}")
			try:
				success, detail = run_case(case, eplus_worker_root, args.quiet)
			except Exception as exc:
				success, detail = False, str(exc)

			if success:
				ok += 1
			else:
				failed += 1
				print(f"Failed case: {case.case_id}")
				print(f"  Manifest: {case.manifest_path} (row {case.row_index})")
				print(f"  Detail: {detail}")
	else:
		payloads = [
			(
				str(c.manifest_path),
				c.row_index,
				c.case_id,
				str(c.idf_path),
				str(c.weather_path),
				str(c.graph_path),
				str(c.energy_npz_path),
				eplus_worker_root,
				args.quiet,
			)
			for c in cases
		]

		done = 0
		if not args.quiet:
			print(f"Parallel mode enabled, effective workers: {workers}")

		with ProcessPoolExecutor(max_workers=workers) as executor:
			future_map = {executor.submit(_run_case_payload, *p): p for p in payloads}
			for future in as_completed(future_map):
				done += 1
				try:
					success, detail, case_id, manifest_path_str, row_index = future.result()
				except Exception as exc:
					success = False
					detail = str(exc)
					case_id = "unknown"
					manifest_path_str = "unknown"
					row_index = -1

				if success:
					ok += 1
				else:
					failed += 1
					print(f"Failed case: {case_id}")
					print(f"  Manifest: {manifest_path_str} (row {row_index})")
					print(f"  Detail: {detail}")

				if not args.quiet:
					core_status = _format_core_status(len(future_map) - done, workers)
					print(
						f"\rProcessed: {done}/{total} | Core status: {core_status}",
						end="",
						flush=True,
					)

		if not args.quiet:
			print()

	print("\nCity refresh finished")
	print(f"  Success: {ok}")
	print(f"  Failed:  {failed}")
	print(f"  Total:   {ok + failed}")

	if resolve_errors and not args.quiet:
		print(f"\nRows skipped before simulation: {len(resolve_errors)}")
		for msg in resolve_errors[:20]:
			print(f"  - {msg}")
		if len(resolve_errors) > 20:
			print(f"  ... and {len(resolve_errors) - 20} more")

	return 0 if failed == 0 else 3


if __name__ == "__main__":
	raise SystemExit(main())
