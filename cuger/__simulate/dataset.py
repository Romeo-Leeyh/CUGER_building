"""Dataset builder for EnergyPlus batch outputs.

This module builds per-case PKL files by combining:
- weather data from EPW/ZIP
- building graph from JSON
- energy data from exported CSV
"""

from __future__ import annotations

import csv
import pickle
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from cuger.graphIO import json_to_graph


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_energy_csv(csv_path: Path) -> Dict[str, List[float]]:
    with csv_path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            return {}

        series: Dict[str, List[float]] = {
            name: [] for name in reader.fieldnames if name and name.lower() != "hour"
        }

        for row in reader:
            for name in series:
                series[name].append(_safe_float(row.get(name, "0")))

    return series


def _read_weather_rows(epw_path: Path) -> tuple[List[str], List[List[str]]]:
    if epw_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(epw_path, "r") as zf:
            epw_members = [n for n in zf.namelist() if n.lower().endswith(".epw")]
            if not epw_members:
                raise ValueError(f"No .epw file in zip: {epw_path}")
            with zf.open(epw_members[0], "r") as fp:
                text = fp.read().decode("utf-8", errors="replace")
        lines = text.splitlines()
    else:
        with epw_path.open("r", encoding="utf-8", errors="replace") as fp:
            lines = fp.read().splitlines()

    header = lines[:8]
    body = [line.split(",") for line in lines[8:] if line.strip()]
    return header, body


def _read_weather_data(epw_path: Path) -> Dict[str, object]:
    header, rows = _read_weather_rows(epw_path)

    parsed = {
        "dry_bulb": [],
        "dew_point": [],
        "relative_humidity": [],
        "global_horizontal_radiation": [],
        "direct_normal_radiation": [],
        "diffuse_horizontal_radiation": [],
        "wind_speed": [],
    }

    for row in rows:
        # EPW columns are 1-based in docs; indices here are 0-based.
        # 7: dry bulb, 8: dew point, 9: RH, 13: GHI, 14: DNI, 15: DHI, 21: wind speed.
        parsed["dry_bulb"].append(_safe_float(row[6] if len(row) > 6 else "0"))
        parsed["dew_point"].append(_safe_float(row[7] if len(row) > 7 else "0"))
        parsed["relative_humidity"].append(_safe_float(row[8] if len(row) > 8 else "0"))
        parsed["global_horizontal_radiation"].append(_safe_float(row[12] if len(row) > 12 else "0"))
        parsed["direct_normal_radiation"].append(_safe_float(row[13] if len(row) > 13 else "0"))
        parsed["diffuse_horizontal_radiation"].append(_safe_float(row[14] if len(row) > 14 else "0"))
        parsed["wind_speed"].append(_safe_float(row[20] if len(row) > 20 else "0"))

    return {
        "source": str(epw_path),
        "header": header,
        "series": parsed,
    }


def _read_job_rows(job_pairs_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with job_pairs_path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(row)
    return rows


def _build_graph_index(graph_dir: Path) -> Dict[str, Path]:
    return {p.stem.lower(): p for p in graph_dir.rglob("*.json") if p.is_file()}


def _find_graph_path(
    graph_index: Dict[str, Path],
    idf_stem: str,
    idf_tag: str,
) -> Optional[Path]:
    if idf_stem.lower() in graph_index:
        return graph_index[idf_stem.lower()]
    if idf_tag.lower() in graph_index:
        return graph_index[idf_tag.lower()]

    needle = idf_stem.lower()
    for stem, path in graph_index.items():
        if needle in stem:
            return path

    return None


def _scan_existing_tags(dataset_dir: Path) -> set[str]:
    if not dataset_dir.exists():
        return set()
    return {p.stem for p in dataset_dir.rglob("*.pkl") if p.is_file()}


def _resolve_weather_path(raw_weather_path: Path, epw_dir: Optional[Path]) -> Optional[Path]:
    if raw_weather_path.exists():
        return raw_weather_path

    if epw_dir is None or (not epw_dir.exists()):
        return None

    by_name = epw_dir / raw_weather_path.name
    if by_name.exists():
        return by_name

    stem = raw_weather_path.stem.lower()
    for p in epw_dir.rglob("*"):
        if p.is_file() and p.stem.lower() == stem and p.suffix.lower() in {".epw", ".zip"}:
            return p

    return None


def build_dataset_from_job_pairs(
    job_pairs_path: Path,
    csv_dir: Path,
    graph_dir: Path,
    dataset_dir: Path,
    epw_dir: Optional[Path] = None,
    skip_existing: bool = True,
    quiet: bool = False,
) -> Dict[str, int]:
    """Build per-job PKL files using completed job metadata.

    Only jobs with status=completed and result=1 are considered.
    Existing PKLs in dataset_dir are scanned first and skipped when requested.
    """

    stats = {
        "total_rows": 0,
        "eligible": 0,
        "built": 0,
        "skipped_existing": 0,
        "skipped_missing_input": 0,
        "failed": 0,
    }

    if not job_pairs_path.exists():
        return stats

    dataset_dir.mkdir(parents=True, exist_ok=True)
    graph_index = _build_graph_index(graph_dir)
    existing_tags = _scan_existing_tags(dataset_dir) if skip_existing else set()

    rows = _read_job_rows(job_pairs_path)
    stats["total_rows"] = len(rows)

    for row in rows:
        status = (row.get("status") or "").strip().lower()
        result = (row.get("result") or "").strip()
        if status != "completed" or result != "1":
            continue

        stats["eligible"] += 1

        job_tag = (row.get("job_tag") or "").strip()
        idf_path_raw = (row.get("idf_path") or "").strip()
        weather_path_raw = (row.get("weather_path") or "").strip()
        idf_tag = (row.get("idf_tag") or "").strip()

        if not job_tag or not idf_path_raw or not weather_path_raw:
            stats["skipped_missing_input"] += 1
            continue

        if skip_existing and job_tag in existing_tags:
            stats["skipped_existing"] += 1
            continue

        idf_path = Path(idf_path_raw)
        weather_path = Path(weather_path_raw)
        resolved_weather_path = _resolve_weather_path(weather_path, epw_dir)
        csv_path = csv_dir / f"{job_tag}.csv"
        idf_stem = idf_path.stem

        graph_path = _find_graph_path(graph_index, idf_stem=idf_stem, idf_tag=idf_tag)

        if not csv_path.exists() or resolved_weather_path is None or graph_path is None:
            stats["skipped_missing_input"] += 1
            continue

        out_path = dataset_dir / f"{job_tag}.pkl"

        try:
            payload = {
                "job_tag": job_tag,
                "idf_path": str(idf_path),
                "weather_path": str(resolved_weather_path),
                "graph_path": str(graph_path),
                "weather": _read_weather_data(resolved_weather_path),
                "building": json_to_graph(str(graph_path)),
                "energy": _read_energy_csv(csv_path),
            }
            with out_path.open("wb") as fp:
                pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
            stats["built"] += 1
            if not quiet:
                print(f"Dataset built: {out_path.name}")
        except Exception as exc:  # noqa: BLE001
            stats["failed"] += 1
            if not quiet:
                print(f"Dataset build failed for {job_tag}: {exc}")

    return stats
