import sys
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
input_dir = Path("tests\examples")  # Relative path - works on all platforms
output_dir = Path("tests\examples_results")  # Relative path - works on all platforms

# For absolute paths on specific systems, use Path() instead of raw strings:
# input_dir = Path("/home/user/data/geo")      # Linux
# input_dir = Path("/Users/user/data/geo")     # macOS
# input_dir = Path("C:/Users/user/data/geo")   # Windows (use forward slashes with Path)

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)


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

    print(f"Processing file: {input_geo_path}")
    print(f"  Model: {modelname}, LOD: {lod}")
    
    # Step 1: Simplify the input geometry based on LOD
    simplified_geo_path = paths["simplified_geo_path"]
    
    # Create output directory if needed
    Path(simplified_geo_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Perform simplification
    print(f"  Step 1: Simplifying geometry (LOD={lod})...")

    ps.simplify_process(input_geo_path, simplified_geo_path, 
                        figure_path=None, lod=lod)
    print(f"    [OK] Simplified geometry saved to: {simplified_geo_path}")

    
    # Step 2: Convexify the simplified geometry
    print(f"  Step 2: Convexifying simplified geometry...")
    try:
        ps.convex_process(
            simplified_geo_path,
            paths["convex_geo_path"],
            paths["figure_convex_path"],
            overlay_geo_path=input_geo_path,
        )
        print(f"    [OK] Convexified geometry saved to: {paths['convex_geo_path']}")

    except Exception as e:
        print(f"    [FAILED] Convexification failed: {e}")
        return False
    
    

    # Step 3: Transform with Moosas (optional, currently commented out)
    # Uncomment the following code to enable Moosas transformation
    try:
        model = Moosas.transform(paths["convex_geo_path"], 
                       solve_overlap=True, 
                      divided_zones=False, 
                      break_wall_horizontal=True, 
                      solve_redundant=True,
                       attach_shading=False,
                       standardize=True) 

        Moosas.saveModel(model, paths["new_geo_path"], save_type="geo")
        Moosas.saveModel(model, paths["new_xml_path"], save_type="xml")
        #Moosas.saveModel(model, paths["new_rdf_path"], save_type="rdf")
        Moosas.saveModel(model, paths["new_idf_path"], save_type="idf")
        print(f"    ✓ Moosas transformation completed")
    
    except Exception as e:
        print(f"    ✗ Moosas transformation failed: {e}")
    
    # Step 4: Generate graph (optional, currently commented out)
    # Uncomment the following code to enable graph generation
    try:
        ps.graph_process(paths["new_geo_path"], paths["new_xml_path"], 
                        paths["output_graph_path"], paths["figure_graph_path"])
        print(f"    ✓ Graph generated")
    except Exception as e:
        print(f"    ✗ Graph generation failed: {e}")

    return True


def main():
    """Main function to process all GEO files with different LOD levels."""
    
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}\n")
    
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
    
    print(f"Found {len(geo_files)} GEO file(s) to process\n")
    
    # Process files with different LOD levels
    lod = "precise"  # Change to 'medium' or 'low' as needed
    processed_count = 0
    skipped_count = 0

        
    for input_geo_path, basename in geo_files:
        if is_file_processed(basename, lod=lod):
            print(f"[SKIPPED] Already processed: {basename}\n")
            skipped_count += 1
            continue

        if process_file(input_geo_path, basename, lod=lod):
            print(f"[OK] Successfully processed: {basename}\n")
            processed_count += 1
        else:
            print(f"[FAILED] Failed to process: {basename}\n")
        

    
    print("=" * 80)
    print(f"Processed: {processed_count}, Skipped: {skipped_count}, Total: {len(geo_files)}")
    print("Processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
