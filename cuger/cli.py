from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PipelineOptions, process_geo_directory, process_geo_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CUGER geometry pipeline")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-s", "--single", help="Path to a single .geo file")
    source.add_argument("-i", "--input-dir", help="Directory containing .geo files")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory")
    parser.add_argument("-l", "--lod", default="precise", choices=["precise", "medium", "low"])
    parser.add_argument("--skip-moosas", action="store_true", help="Only run simplify + convexify")
    parser.add_argument("--skip-graph", action="store_true", help="Skip graph generation")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["geo", "xml", "idf"],
        choices=["geo", "xml", "idf", "rdf"],
        help="Output formats when Moosas export is enabled",
    )
    parser.add_argument("--no-convex-figure", action="store_true", help="Do not save convex figures")
    parser.add_argument("--no-graph-figure", action="store_true", help="Do not save graph figures")
    parser.add_argument("--non-recursive", action="store_true", help="Do not scan subdirectories")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    options = PipelineOptions(
        lod=args.lod,
        run_moosas=not args.skip_moosas,
        generate_graph=not args.skip_graph,
        save_formats=tuple(args.formats),
        convex_figure=not args.no_convex_figure,
        graph_figure=not args.no_graph_figure,
    )

    if args.single:
        result = process_geo_file(
            args.single,
            args.output_dir,
            modelname=Path(args.single).stem,
            options=options,
        )
        print(f"Processed {result['modelname']}")
        return 0

    results = process_geo_directory(
        args.input_dir,
        args.output_dir,
        recursive=not args.non_recursive,
        options=options,
    )
    print(f"Processed {len(results)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())