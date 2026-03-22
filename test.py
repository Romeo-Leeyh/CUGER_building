import sys
import os
import io
import contextlib
import argparse
import time
from pathlib import Path

# Ensure workspace root is importable when running this file directly
CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from cuger.__transform import process as ps
import moosas.MoosasPy as Moosas

# Define input/output directories (use pathlib for cross-platform compatibility)
# Change these paths according to your local environment
input_dir = Path("/mnt/z/lyh/SRT/EVOMASS+/geo")  # Relative path - works on all platforms
output_dir = Path("/mnt/z/lyh/SRT/EVOMASS+/")  # Relative path - works on all platforms

# For absolute paths on specific systems, use Path() instead of raw strings:
# input_dir = Path("/home/user/data/geo")      # Linux
# input_dir = Path("/Users/user/data/geo")     # macOS
# input_dir = Path("C:/Users/user/data/geo")   # Windows (use forward slashes with Path)

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)


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


def is_file_processed(modelname, lod="precise"):
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


def process_file(input_geo_path, modelname, lod="precise"):
    """
    Process a single GEO file through the simplification and convexification pipeline.
    
    Parameters:
    -----------
    input_geo_path : str
        Path to input GEO file
    modelname : str
        Name of the model (used for output file naming)
    lod : str
        Level of detail: 'precise', 'medium', or 'low'
    """
    paths = ps.get_output_paths(modelname, str(output_dir), lod=lod)

    # Step 1: Simplify the input geometry based on LOD
    simplified_geo_path = paths["simplified_geo_path"]
    
    # Create output directory if needed
    Path(simplified_geo_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Perform simplification
    ps.simplify_process(input_geo_path, simplified_geo_path, 
                        figure_path=None, lod=lod)

    
    # Step 2: Convexify the simplified geometry
    try:
        ps.convex_process(
            simplified_geo_path,
            paths["convex_geo_path"],
            paths["figure_convex_path"],
            overlay_geo_path=input_geo_path,
        )

    except Exception as e:
        _ = e
        return False
    
    

    # Step 3: Transform with Moosas (optional, currently commented out)
    # Uncomment the following code to enable Moosas transformation
    try:
        with suppress_output():
            model = Moosas.transform(paths["convex_geo_path"], 
                           solve_overlap=True, 
                          divided_zones=False, 
                          break_wall_horizontal=True, 
                          solve_redundant=True,
                           attach_shading=False,
                           standardize=True,
                           stdout=io.StringIO())

            Moosas.saveModel(model, paths["new_geo_path"], save_type="geo")
            Moosas.saveModel(model, paths["new_xml_path"], save_type="xml")
            #Moosas.saveModel(model, paths["new_rdf_path"], save_type="rdf")
            Moosas.saveModel(model, paths["new_idf_path"], save_type="idf")
    
    except Exception as e:
        _ = e
    
    # Step 4: Generate graph (optional, currently commented out)
    # Uncomment the following code to enable graph generation
    try:
        ps.graph_process(paths["new_geo_path"], paths["new_xml_path"], 
                        paths["output_graph_path"], paths["figure_graph_path"])
    except Exception as e:
        _ = e

    return True


def parse_args():
    """Parse optional CLI args for worker-based sharding."""
    parser = argparse.ArgumentParser(
        description="Process GEO files through CUGER pipeline."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Total number of workers used for static file sharding.",
    )
    parser.add_argument(
        "--worker-index",
        type=int,
        default=0,
        help="Current worker index in [0, workers-1].",
    )
    parser.add_argument(
        "--lod",
        type=str,
        default="precise",
        choices=["precise", "medium", "low"],
        help="Level of detail for simplification.",
    )
    return parser.parse_args()


def main():
    """Main function to process all GEO files with different LOD levels."""
    args = parse_args()
    workers = args.workers
    worker_index = args.worker_index

    if workers < 1:
        print(f"Error: --workers must be >= 1, got {workers}")
        return
    if worker_index < 0 or worker_index >= workers:
        print(
            f"Error: --worker-index must be in [0, {workers - 1}], got {worker_index}"
        )
        return
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    # Get all GEO files using pathlib for cross-platform compatibility
    geo_files = []
    for geo_filepath in sorted(input_dir.rglob('*.geo')):  # Recursive glob for all .geo files
        relative_path = geo_filepath.relative_to(input_dir)
        # Build modelname from relative path in a cross-platform way
        basename = relative_path.with_suffix("").as_posix().replace("/", "_")
        geo_files.append((geo_filepath, basename))

    if not geo_files:
        print(f"No GEO files found in {input_dir}")
        return
    
    geo_files = [
        pair for idx, pair in enumerate(geo_files) if idx % workers == worker_index
    ]
    total_assigned = len(geo_files)
    
    # Process files with different LOD levels
    lod = args.lod
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    # Emit initial progress so the parallel runner can display global totals early.
    print(
        f"progress worker={worker_index} completed=0/{total_assigned} "
        f"skipped=0 processed=0 failed=0 elapsed_s=0.0",
        flush=True,
    )

    start_time = time.time()

        
    for input_geo_path, basename in geo_files:
        if is_file_processed(basename, lod=lod):
            skipped_count += 1
            completed = skipped_count + processed_count + failed_count
            elapsed_s = time.time() - start_time
            print(
                f"progress worker={worker_index} completed={completed}/{total_assigned} "
                f"skipped={skipped_count} processed={processed_count} failed={failed_count} "
                f"elapsed_s={elapsed_s:.1f} current={basename}",
                flush=True,
            )
            continue

        if process_file(input_geo_path, basename, lod=lod):
            processed_count += 1
        else:
            failed_count += 1

        completed = skipped_count + processed_count + failed_count
        elapsed_s = time.time() - start_time
        print(
            f"progress worker={worker_index} completed={completed}/{total_assigned} "
            f"skipped={skipped_count} processed={processed_count} failed={failed_count} "
            f"elapsed_s={elapsed_s:.1f} current={basename}",
            flush=True,
        )

    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
        core_label = ",".join(str(c) for c in affinity)
    else:
        core_label = "N/A"

    print(
        f"core={core_label} assigned={total_assigned} "
        f"already_processed={skipped_count} processed_now={processed_count} failed={failed_count}"
    )


if __name__ == "__main__":
    main()
