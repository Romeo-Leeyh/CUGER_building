"""Build packed dataset in demo.py structured format.

This module reads completed jobs from job_pairs.csv and writes a structured
pack directory with:
- weather/*.npz (deduplicated by hash)
- building/*.npz (deduplicated by hash)
- energy/*.npz (per sample)
- manifest.csv
- meta.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from cuger.graphIO import json_to_graph

try:
    from prep import dataload as dl
except Exception:  # noqa: BLE001
    dl = None


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_job_rows(job_pairs_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with job_pairs_path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(row)
    return rows


def _build_graph_index(graph_dir: Path) -> Dict[str, Path]:
    return {p.stem.lower(): p for p in graph_dir.rglob("*.json") if p.is_file()}


def _find_graph_path(graph_index: Dict[str, Path], idf_stem: str, idf_tag: str) -> Optional[Path]:
    if idf_stem.lower() in graph_index:
        return graph_index[idf_stem.lower()]
    if idf_tag.lower() in graph_index:
        return graph_index[idf_tag.lower()]

    needle = idf_stem.lower()
    for stem, path in graph_index.items():
        if needle in stem:
            return path
    return None


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


def _read_weather_rows(epw_path: Path) -> Tuple[List[str], List[List[str]]]:
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


def _read_weather_fallback(epw_path: Path) -> pd.DataFrame:
    _header, rows = _read_weather_rows(epw_path)

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
        parsed["dry_bulb"].append(_safe_float(row[6] if len(row) > 6 else "0"))
        parsed["dew_point"].append(_safe_float(row[7] if len(row) > 7 else "0"))
        parsed["relative_humidity"].append(_safe_float(row[8] if len(row) > 8 else "0"))
        parsed["global_horizontal_radiation"].append(_safe_float(row[12] if len(row) > 12 else "0"))
        parsed["direct_normal_radiation"].append(_safe_float(row[13] if len(row) > 13 else "0"))
        parsed["diffuse_horizontal_radiation"].append(_safe_float(row[14] if len(row) > 14 else "0"))
        parsed["wind_speed"].append(_safe_float(row[20] if len(row) > 20 else "0"))

    return pd.DataFrame(parsed)


def _read_energy_npz(npz_path: Path) -> pd.DataFrame:
    with np.load(npz_path, allow_pickle=True) as data:
        values = data["values"]
        columns = data["columns"]
    cols = [str(c) for c in columns.tolist()]
    return pd.DataFrame(values, columns=cols)


def _normalize_table(table_obj) -> pd.DataFrame:
    if isinstance(table_obj, pd.DataFrame):
        return table_obj
    if isinstance(table_obj, dict):
        try:
            return pd.DataFrame(table_obj)
        except Exception:
            return pd.DataFrame([table_obj])
    if isinstance(table_obj, np.ndarray):
        return pd.DataFrame(table_obj)
    if isinstance(table_obj, (list, tuple)):
        return pd.DataFrame(table_obj)
    return pd.DataFrame([table_obj])


def _table_to_values_and_columns(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    cols = np.asarray([str(c) for c in df.columns], dtype=str)
    try:
        values = df.to_numpy(dtype=np.float32, copy=True)
    except Exception:
        values = df.astype(str).to_numpy(dtype=object, copy=True)
    return values, cols


def _hash_array(h: "hashlib._Hash", arr: np.ndarray) -> None:
    arr_c = np.ascontiguousarray(arr)
    h.update(str(arr_c.dtype).encode("utf-8"))
    h.update(str(arr_c.shape).encode("utf-8"))
    h.update(arr_c.tobytes())


def _hash_weather(weather_df: pd.DataFrame) -> str:
    values, cols = _table_to_values_and_columns(weather_df)
    h = hashlib.blake2b(digest_size=16)
    _hash_array(h, values)
    h.update("|".join(cols.tolist()).encode("utf-8"))
    return h.hexdigest()


def _normalize_building_from_graph(building_graph: nx.Graph, energy_columns=None) -> Dict[str, np.ndarray]:
    energy_columns = [str(c) for c in (energy_columns or [])]
    energy_lut = {c.lower(): c for c in energy_columns}

    nodes_dict = dict(building_graph.nodes(data=True))

    face_id_map: Dict[object, int] = {}
    space_id_map: Dict[object, int] = {}
    face_feats: List[List[float]] = []
    space_feats: List[List[float]] = []

    valid_energy_spaces: List[str] = []
    space_nodes: List[Tuple[object, str]] = []
    for nid, attrs in nodes_dict.items():
        if attrs.get("node_type") == "space":
            nid_str = str(nid)
            if energy_lut:
                mapped = energy_lut.get(nid_str.lower())
                if mapped is not None:
                    valid_energy_spaces.append(mapped)
                    space_nodes.append((nid, mapped))
            else:
                valid_energy_spaces.append(nid_str)
                space_nodes.append((nid, nid_str))

    valid_space_node_ids = {nid for nid, _ in space_nodes}

    for nid, attrs in nodes_dict.items():
        node_type = attrs.get("node_type")
        if node_type == "face":
            p = attrs.get("face_params", {}) or {}
            s = np.asarray(p.get("s", []), dtype=np.float32).reshape(-1)
            n = np.asarray(p.get("n", []), dtype=np.float32).reshape(-1)
            t_mapped = 1.0 if p.get("t") in ["window", "airwall"] else 0.0
            feat = np.concatenate([s, n, np.array([t_mapped], dtype=np.float32)])
            face_id_map[nid] = len(face_feats)
            face_feats.append(feat.tolist())
        elif node_type == "space" and nid in valid_space_node_ids:
            p = attrs.get("space_params", {}) or {}
            s = np.asarray(p.get("s", []), dtype=np.float32).reshape(-1)
            if s.size == 0:
                a = p.get("a", 0.0)
                s = np.array([a], dtype=np.float32)
            space_id_map[nid] = len(space_feats)
            space_feats.append(s.tolist())

    ff_edges: List[List[int]] = []
    sf_edges: List[List[int]] = []
    sf_edge_attr: List[List[float]] = []

    for u, v, attrs in building_graph.edges(data=True):
        ut = nodes_dict.get(u, {}).get("node_type")
        vt = nodes_dict.get(v, {}).get("node_type")

        if ut == "face" and vt == "face" and u in face_id_map and v in face_id_map:
            fu, fv = face_id_map[u], face_id_map[v]
            ff_edges.append([fu, fv])
            ff_edges.append([fv, fu])
            continue

        layer_flag = 1.0 if attrs.get("layer", 0) != 0 else 0.0
        if ut == "face" and vt == "space" and u in face_id_map and v in space_id_map:
            sf_edges.append([face_id_map[u], space_id_map[v]])
            sf_edge_attr.append([layer_flag])
        elif ut == "space" and vt == "face" and v in face_id_map and u in space_id_map:
            sf_edges.append([face_id_map[v], space_id_map[u]])
            sf_edge_attr.append([layer_flag])

    return {
        "ff_edges": np.asarray(ff_edges, dtype=np.int32),
        "sf_edges": np.asarray(sf_edges, dtype=np.int32),
        "sf_edge_attr": np.asarray(sf_edge_attr, dtype=np.float32),
        "face_feats": np.asarray(face_feats, dtype=np.float32),
        "space_feats": np.asarray(space_feats, dtype=np.float32),
        "valid_energy_spaces": np.asarray(valid_energy_spaces, dtype=str),
    }


def _normalize_building(building, energy_columns=None) -> Dict[str, np.ndarray]:
    if isinstance(building, dict):
        if "ff_edges" in building and "sf_edges" in building:
            return {
                "ff_edges": np.asarray(building.get("ff_edges", []), dtype=np.int32),
                "sf_edges": np.asarray(building.get("sf_edges", []), dtype=np.int32),
                "sf_edge_attr": np.asarray(building.get("sf_edge_attr", []), dtype=np.float32),
                "face_feats": np.asarray(building.get("face_feats", []), dtype=np.float32),
                "space_feats": np.asarray(building.get("space_feats", []), dtype=np.float32),
                "valid_energy_spaces": np.asarray(building.get("valid_energy_spaces", []), dtype=str),
            }

    if isinstance(building, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        return _normalize_building_from_graph(building, energy_columns=energy_columns)

    raise TypeError(f"Unsupported building type: {type(building)}")


def _hash_building(norm_building: Dict[str, np.ndarray]) -> str:
    h = hashlib.blake2b(digest_size=16)
    for key in ["ff_edges", "sf_edges", "sf_edge_attr", "face_feats", "space_feats", "valid_energy_spaces"]:
        h.update(key.encode("utf-8"))
        _hash_array(h, norm_building[key])
    return h.hexdigest()


def _save_weather(path: Path, weather_df: pd.DataFrame) -> None:
    values, cols = _table_to_values_and_columns(weather_df)
    np.savez_compressed(path, values=values, columns=cols)


def _save_building(path: Path, norm_building: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(
        path,
        ff_edges=norm_building["ff_edges"],
        sf_edges=norm_building["sf_edges"],
        sf_edge_attr=norm_building["sf_edge_attr"],
        face_feats=norm_building["face_feats"],
        space_feats=norm_building["space_feats"],
        valid_energy_spaces=norm_building["valid_energy_spaces"],
    )


def _save_energy(path: Path, energy_df: pd.DataFrame) -> None:
    values, cols = _table_to_values_and_columns(energy_df)
    np.savez_compressed(path, values=values, columns=cols)


def _sanitize_name(name: str) -> str:
    """Normalize a name into a filesystem-safe stem while preserving readability."""
    text = (name or "").strip().replace("/", "_").replace("\\", "_")
    for ch in [":", "*", "?", '"', "<", ">", "|"]:
        text = text.replace(ch, "_")
    return text or "unknown"


def _build_case_payload(weather_path: Path, graph_path: Path, energy_path: Path) -> Dict[str, object]:
    energy_data = _read_energy_npz(energy_path)
    if dl is not None:
        weather_data = dl.input_weather(str(weather_path))
        building_graph = dl.input_building(str(graph_path))
        building_data = dl.align_and_reorder_graph_by_energy(building_graph, energy_data)
        valid_spaces = building_data.get("valid_energy_spaces") if isinstance(building_data, dict) else None
        if valid_spaces is not None:
            try:
                energy_data = energy_data[valid_spaces]
            except Exception:  # noqa: BLE001
                pass
        return {"weather": weather_data, "building": building_data, "energy": energy_data}

    return {
        "weather": _read_weather_fallback(weather_path),
        "building": json_to_graph(str(graph_path)),
        "energy": energy_data,
    }


def build_packed_dataset_from_job_pairs(
    job_pairs_path: Path,
    energy_dir: Path,
    graph_dir: Path,
    pack_dir: Path,
    epw_dir: Optional[Path] = None,
    skip_existing: bool = True,
    quiet: bool = False,
) -> Dict[str, int]:
    """Build packed dataset directory in demo.py format."""

    stats = {
        "total_rows": 0,
        "eligible": 0,
        "built": 0,
        "skipped_existing": 0,
        "skipped_missing_input": 0,
        "failed": 0,
        "weather_unique": 0,
        "building_unique": 0,
    }

    if not job_pairs_path.exists():
        return stats

    weather_dir = pack_dir / "weather"
    building_dir = pack_dir / "building"
    energy_dir = pack_dir / "energy"
    weather_dir.mkdir(parents=True, exist_ok=True)
    building_dir.mkdir(parents=True, exist_ok=True)
    energy_dir.mkdir(parents=True, exist_ok=True)

    graph_index = _build_graph_index(graph_dir)
    existing_samples = {p.stem for p in energy_dir.glob("*.npz")} if skip_existing else set()

    weather_seen = {p.stem for p in weather_dir.glob("*.npz")}
    building_seen = {p.stem for p in building_dir.glob("*.npz")}

    manifest_path = pack_dir / "manifest.csv"
    existing_manifest_rows: List[Dict[str, str]] = []
    manifest_sample_ids = set()
    if manifest_path.exists():
        with manifest_path.open("r", newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                sample_id = (row.get("sample_id") or "").strip()
                if sample_id:
                    manifest_sample_ids.add(sample_id)
                existing_manifest_rows.append(row)

    rows = _read_job_rows(job_pairs_path)
    stats["total_rows"] = len(rows)

    new_manifest_rows: List[Dict[str, str]] = []

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

        sample_id = job_tag
        if sample_id in manifest_sample_ids:
            stats["skipped_existing"] += 1
            continue
        if skip_existing and (sample_id in existing_samples or sample_id in manifest_sample_ids):
            stats["skipped_existing"] += 1
            continue

        idf_path = Path(idf_path_raw)
        weather_path = Path(weather_path_raw)
        resolved_weather_path = _resolve_weather_path(weather_path, epw_dir)
        npz_path = energy_dir / f"{job_tag}.npz"
        idf_stem = idf_path.stem
        graph_path = _find_graph_path(graph_index, idf_stem=idf_stem, idf_tag=idf_tag)
        energy_path = npz_path

        if not energy_path.exists() or resolved_weather_path is None or graph_path is None:
            stats["skipped_missing_input"] += 1
            continue

        try:
            payload = _build_case_payload(
                weather_path=resolved_weather_path,
                graph_path=graph_path,
                energy_path=energy_path,
            )
            weather_df = _normalize_table(payload["weather"])
            energy_df = _normalize_table(payload["energy"])
            norm_building = _normalize_building(payload["building"], energy_columns=list(energy_df.columns))

            # Use semantic IDs instead of hash-only names so files align with
            # data naming: sample_id = building__weather.
            weather_id = _sanitize_name(weather_path.stem)
            building_id = _sanitize_name(graph_path.stem)

            weather_path_out = weather_dir / f"{weather_id}.npz"
            if weather_id not in weather_seen and not weather_path_out.exists():
                _save_weather(weather_path_out, weather_df)
                weather_seen.add(weather_id)
            else:
                weather_seen.add(weather_id)

            building_path_out = building_dir / f"{building_id}.npz"
            if building_id not in building_seen and not building_path_out.exists():
                _save_building(building_path_out, norm_building)
                building_seen.add(building_id)
            else:
                building_seen.add(building_id)

            energy_path_out = energy_dir / f"{sample_id}.npz"
            _save_energy(energy_path_out, energy_df)

            new_manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "source_job_tag": job_tag,
                    "weather_id": weather_id,
                    "building_id": building_id,
                    "energy_file": energy_path_out.name,
                    "n_steps": str(energy_df.shape[0]),
                    "n_spaces": str(energy_df.shape[1]),
                }
            )
            manifest_sample_ids.add(sample_id)
            stats["built"] += 1
            if not quiet:
                print(f"Packed sample built: {sample_id}")
        except Exception as exc:  # noqa: BLE001
            stats["failed"] += 1
            if not quiet:
                print(f"Packed sample failed for {job_tag}: {exc}")

    manifest_fields = [
        "sample_id",
        "source_job_tag",
        "weather_id",
        "building_id",
        "energy_file",
        "n_steps",
        "n_spaces",
    ]
    write_header = (not manifest_path.exists()) or manifest_path.stat().st_size == 0
    with manifest_path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=manifest_fields)
        if write_header:
            writer.writeheader()
        writer.writerows(new_manifest_rows)

    stats["weather_unique"] = len(weather_seen)
    stats["building_unique"] = len(building_seen)

    meta = {
        "format_version": "structured_v1",
        "source": "job_pairs",
        "downcast": {
            "weather": "float32",
            "energy": "float32",
            "face_feats": "float32",
            "space_feats": "float32",
            "ff_edges": "int32",
            "sf_edges": "int32",
            "sf_edge_attr": "float32",
        },
        "eligible": stats["eligible"],
        "built": stats["built"],
        "weather_unique": stats["weather_unique"],
        "building_unique": stats["building_unique"],
    }
    with (pack_dir / "meta.json").open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)

    return stats


def build_packed_dataset_from_manifest(
    manifest_path: Path,
    energy_dir: Path,
    graph_dir: Path,
    pack_dir: Path,
    epw_dir: Optional[Path] = None,
    skip_existing: bool = True,
    quiet: bool = False,
) -> Dict[str, int]:
    """Build packed dataset using a manifest CSV.

    The manifest CSV uses the same required columns as legacy job_pairs.csv,
    so this is a naming-level replacement that keeps data compatibility.
    """

    return build_packed_dataset_from_job_pairs(
        job_pairs_path=manifest_path,
        energy_dir=energy_dir,
        graph_dir=graph_dir,
        pack_dir=pack_dir,
        epw_dir=epw_dir,
        skip_existing=skip_existing,
        quiet=quiet,
    )
