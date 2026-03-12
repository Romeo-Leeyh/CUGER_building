from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable

from .__transform.process import (
    convex_process,
    get_output_paths,
    graph_process,
    simplify_process,
)

VALID_LODS = {"precise", "medium", "low"}
DEFAULT_SAVE_FORMATS = ("geo", "xml", "idf")


@dataclass(slots=True)
class PipelineOptions:
    lod: str = "precise"
    run_moosas: bool = True
    generate_graph: bool = True
    save_formats: tuple[str, ...] = DEFAULT_SAVE_FORMATS
    solve_overlap: bool = True
    divided_zones: bool = False
    break_wall_horizontal: bool = True
    solve_redundant: bool = True
    attach_shading: bool = False
    standardize: bool = True
    convex_figure: bool = True
    graph_figure: bool = True

    def normalized_formats(self) -> tuple[str, ...]:
        formats = tuple(fmt.lower() for fmt in self.save_formats)
        invalid = sorted(set(formats) - {"geo", "xml", "idf", "rdf"})
        if invalid:
            raise ValueError(f"Unsupported save formats: {', '.join(invalid)}")
        return formats

    def validate(self) -> None:
        if self.lod not in VALID_LODS:
            raise ValueError(f"lod must be one of: {', '.join(sorted(VALID_LODS))}")
        formats = self.normalized_formats()
        if self.generate_graph and ("geo" not in formats or "xml" not in formats):
            raise ValueError("Graph generation requires both 'geo' and 'xml' in save_formats")
        if self.generate_graph and not self.run_moosas:
            raise ValueError("Graph generation requires run_moosas=True")


def _load_moosas_module(moosas_module: Any | None = None) -> Any:
    if moosas_module is not None:
        return moosas_module

    try:
        return import_module("moosas.MoosasPy")
    except ImportError as exc:
        raise ImportError(
            "Moosas is required for model export. Install Moosas or call the pipeline "
            "with PipelineOptions(run_moosas=False)."
        ) from exc


def _coerce_options(options: PipelineOptions | None = None, **kwargs: Any) -> PipelineOptions:
    if options is None:
        options = PipelineOptions(**kwargs)
    elif kwargs:
        raise ValueError("Pass either a PipelineOptions instance or keyword overrides, not both")

    options.validate()
    return options


def process_geo_file(
    input_geo_path: str | Path,
    output_dir: str | Path,
    modelname: str | None = None,
    *,
    options: PipelineOptions | None = None,
    moosas_module: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the main CUGER pipeline for a single `.geo` file."""
    options = _coerce_options(options, **kwargs)

    input_path = Path(input_geo_path)
    output_path = Path(output_dir)
    modelname = modelname or input_path.stem
    paths = get_output_paths(modelname, str(output_path), lod=options.lod)

    simplify_process(
        str(input_path),
        paths["simplified_geo_path"],
        lod=options.lod,
    )

    convex_process(
        paths["simplified_geo_path"],
        paths["convex_geo_path"],
        paths["figure_convex_path"] if options.convex_figure else None,
        overlay_geo_path=str(input_path),
    )

    moosas_used = False
    if options.run_moosas:
        moosas = _load_moosas_module(moosas_module)
        model = moosas.transform(
            paths["convex_geo_path"],
            solve_overlap=options.solve_overlap,
            divided_zones=options.divided_zones,
            break_wall_horizontal=options.break_wall_horizontal,
            solve_redundant=options.solve_redundant,
            attach_shading=options.attach_shading,
            standardize=options.standardize,
        )

        for save_format in options.normalized_formats():
            moosas.saveModel(model, paths[f"new_{save_format}_path"], save_type=save_format)
        moosas_used = True

    if options.generate_graph:
        graph_process(
            paths["new_geo_path"],
            paths["new_xml_path"],
            paths["output_graph_path"],
            paths["figure_graph_path"] if options.graph_figure else None,
        )

    return {
        "modelname": modelname,
        "input_geo_path": str(input_path),
        "output_dir": str(output_path),
        "paths": paths,
        "lod": options.lod,
        "moosas_used": moosas_used,
        "graph_generated": options.generate_graph,
    }


def process_geo_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    recursive: bool = True,
    options: PipelineOptions | None = None,
    moosas_module: Any | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run the main CUGER pipeline for all `.geo` files in a directory."""
    options = _coerce_options(options, **kwargs)

    root = Path(input_dir)
    pattern = "**/*.geo" if recursive else "*.geo"
    geo_files = sorted(root.glob(pattern))

    results = []
    for geo_file in geo_files:
        modelname = str(geo_file.relative_to(root).with_suffix("")).replace("\\", "_").replace("/", "_")
        results.append(
            process_geo_file(
                geo_file,
                output_dir,
                modelname=modelname,
                options=options,
                moosas_module=moosas_module,
            )
        )

    return results