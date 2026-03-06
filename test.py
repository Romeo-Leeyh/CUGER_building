import os
import sys

from cuger.__transform import process as ps
#import moosas.MoosasPy as Moosas

# Define input/output directories
input_dir = "tests/examples"
output_dir = "tests/example_results"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


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
    paths = ps.get_output_paths(modelname, output_dir, lod=lod)

    print(f"Processing file: {input_geo_path}")
    print(f"  Model: {modelname}, LOD: {lod}")
    
    # Step 1: Simplify the input geometry based on LOD
    simplified_geo_path = paths["simplified_geo_path"]
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(simplified_geo_path), exist_ok=True)
    
    # Perform simplification
    print(f"  Step 1: Simplifying geometry (LOD={lod})...")
    try:
        ps.simplify_process(input_geo_path, simplified_geo_path, 
                           figure_path=None, lod=lod)
        print(f"    [OK] Simplified geometry saved to: {simplified_geo_path}")
    except Exception as e:
        print(f"    [FAILED] Simplification failed: {e}")
        return False
    
    # Step 2: Convexify the simplified geometry
    print(f"  Step 2: Convexifying simplified geometry...")
    try:
        ps.convex_process(simplified_geo_path, paths["convex_geo_path"], 
                         paths["figure_convex_path"])
        print(f"    [OK] Convexified geometry saved to: {paths['convex_geo_path']}")
    except Exception as e:
        print(f"    [FAILED] Convexification failed: {e}")
        return False

    # Step 3: Transform with Moosas (optional, currently commented out)
    # Uncomment the following code to enable Moosas transformation
    # try:
    #     model = Moosas.transform(paths["convex_geo_path"], 
    #                     solve_overlap=True, 
    #                     divided_zones=False, 
    #                     break_wall_horizontal=True, 
    #                     solve_redundant=True,
    #                     attach_shading=False,
    #                     standardize=True)
    #     
    #     Moosas.saveModel(model, paths["new_geo_path"], save_type="geo")
    #     Moosas.saveModel(model, paths["new_xml_path"], save_type="xml")
    #     Moosas.saveModel(model, paths["new_rdf_path"], save_type="rdf")
    #     Moosas.saveModel(model, paths["new_idf_path"], save_type="idf")
    #     print(f"    ✓ Moosas transformation completed")
    # except Exception as e:
    #     print(f"    ✗ Moosas transformation failed: {e}")
    
    # Step 4: Generate graph (optional, currently commented out)
    # Uncomment the following code to enable graph generation
    # try:
    #     ps.graph_process(paths["new_geo_path"], paths["new_xml_path"], 
    #                     paths["output_graph_path"], paths["figure_graph_path"])
    #     print(f"    ✓ Graph generated")
    # except Exception as e:
    #     print(f"    ✗ Graph generation failed: {e}")

    return True


def main():
    """Main function to process all GEO files with different LOD levels."""
    
    print("=" * 80)
    print("BuildingConvex - Three-Level LOD Processing Pipeline")
    print("=" * 80)
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    # Get all GEO files
    geo_files = []
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.geo'):
                input_geo_path = os.path.join(dirpath, filename).replace('\\', '/')
                relative_path = os.path.relpath(input_geo_path, input_dir)
                basename = os.path.splitext(relative_path)[0].replace('\\', '_')
                geo_files.append((input_geo_path, basename))
    
    if not geo_files:
        print(f"No GEO files found in {input_dir}")
        return
    
    print(f"Found {len(geo_files)} GEO file(s) to process\n")
    
    # Process files with different LOD levels
    lod = "precise"  # Change to "medium" or "low" as needed

        
    for input_geo_path, basename in geo_files:
        try:
            if process_file(input_geo_path, basename, lod=lod):
                print(f"[OK] Successfully processed: {basename}\n")
            else:
                print(f"[FAILED] Failed to process: {basename}\n")
        except Exception as e:
            print(f"[ERROR] Error processing {basename}: {str(e)}\n")
    
    
    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
