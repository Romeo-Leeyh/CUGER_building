"""Batch-run EnergyPlus simulations and export SQL outputs to CSV.
"""

import argparse
import tempfile
from pathlib import Path

from cuger.__simulate.simulate import locate_eplus, run_single_simulation  
from cuger.__simulate.sqlread import SQLReader  


DEFAULT_IDF_DIR = Path("data/new_idf")
DEFAULT_EPW_DIR = Path("data/weather")
DEFAULT_CSV_DIR = Path("data/csv_outputs")


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
	parser.add_argument("--quiet", action="store_true", help="Reduce log output")
	return parser


def main() -> int:
	args = build_parser().parse_args()

	idf_dir = Path(args.idf_dir).expanduser().resolve()
	epw_dir = Path(args.epw_dir).expanduser().resolve()
	csv_dir = Path(args.csv_dir).expanduser().resolve()
	csv_dir.mkdir(parents=True, exist_ok=True)

	idf_files = _collect_files(idf_dir, (".idf",))
	weather_files = _collect_files(epw_dir, (".epw", ".zip"))

	if not idf_files:
		print(f"No IDF files found in: {idf_dir}")
		return 1
	if not weather_files:
		print(f"No weather files (.epw/.zip) found in: {epw_dir}")
		return 1

	try:
		exe_path, idd_path = locate_eplus(args.eplus_root)
		if not args.quiet:
			print(f"EnergyPlus executable: {exe_path}")
			print(f"EnergyPlus IDD: {idd_path}")
	except Exception as exc:
		print(f"Failed to locate EnergyPlus: {exc}")
		return 1

	total = len(idf_files) * len(weather_files)
	done = 0
	ok = 0
	failed = 0

	for idf_path in idf_files:
		for weather_path in weather_files:
			done += 1
			idf_tag = _build_tag(idf_path, idf_dir)
			weather_tag = _build_tag(weather_path, epw_dir)
			tag = f"{idf_tag}__{weather_tag}"
			print(f"[{done}/{total}] Running: {tag}")

			# Each run gets an isolated temporary folder and is cleaned up immediately.
			with tempfile.TemporaryDirectory(prefix="cuger_ep_run_") as run_tmp_dir:
				success, sql_path = run_single_simulation(
					idf_path=str(idf_path),
					epw_path=str(weather_path),
					output_dir=run_tmp_dir,
					modify_outputs=True,
					output_variables=None,
					diagnostics=None,
					verbose=not args.quiet,
				)

				if not _valid_sql(success, sql_path):
					failed += 1
					print(f"  FAIL: simulation failed for {tag}")
					continue

				try:
					reader = SQLReader(sql_path)
					out_csv = csv_dir / f"{tag}.csv"
					reader.export_external_loads_csv(str(out_csv))
					ok += 1
					print(f"  OK: {out_csv}")
				except Exception as exc:
					failed += 1
					print(f"  FAIL: SQL->CSV failed for {tag}: {exc}")

	print("\nBatch finished")
	print(f"  Success: {ok}")
	print(f"  Failed:  {failed}")
	return 0 if failed == 0 else 2


if __name__ == "__main__":
	raise SystemExit(main())
