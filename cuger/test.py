import os, sys, time

import __transform.process as ps

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if main_dir not in sys.path:
    sys.path.append(main_dir)

import moosas.MoosasPy as Moosas
# moosas package is independent of cuger, so the import path is moosas.MoosasPy

input = "tests/examples"
output = "tests/example_results"

def process_file(input_geo_path, modelname, lod="medium"):
    paths = ps.get_output_paths(modelname, output)

    print(f"Processing file: {input_geo_path}, basename: {modelname}, LOD: {lod}")
    
    # Step 1: Simplify the input geometry based on LOD
    # Generate simplified geometry file path
    simplified_geo_path = os.path.join(output, "new_geo", f"{modelname}_simplified_{lod}.geo")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(simplified_geo_path), exist_ok=True)
    
    # Perform simplification
    print(f"  Step 1: Simplifying geometry (LOD={lod})...")
    ps.simplify_process(input_geo_path, simplified_geo_path, 
                       figure_path=None, lod=lod)
    print(f"  Simplified geometry saved to: {simplified_geo_path}")
    
    # Step 2: Convexify the simplified geometry
    print(f"  Step 2: Convexifying simplified geometry...")
    ps.convex_process(simplified_geo_path, paths["convex_geo_path"], 
                     paths["figure_convex_path"])
    print(f"  Convexified geometry saved to: {paths['convex_geo_path']}")

    # Step 3: Transform with Moosas (commented out for now)
    # model = Moosas.transform(paths["convex_geo_path"], 
    #                 solve_overlap=True, 
    #                 divided_zones=False, 
    #                 break_wall_horizontal=True, 
    #                 solve_redundant=True,
    #                 attach_shading=False,
    #                 standardize=True)
    
    # Moosas.saveModel(model, paths["new_geo_path"], save_type="geo")
    # Moosas.saveModel(model, paths["new_xml_path"], save_type="xml")
    # Moosas.saveModel(model, paths["new_rdf_path"], save_type="rdf")
    # Moosas.saveModel(model, paths["new_idf_path"], save_type="idf")
    
    # Step 4: Generate graph (commented out for now)
    # ps.graph_process(paths["new_geo_path"], paths["new_xml_path"], 
    #                 paths["output_graph_path"], paths["figure_graph_path"])

    

# Main processing loop - test with different LOD levels
lod_levels = ["precise", "medium", "low"]

for lod in lod_levels:
    print(f"\n{'='*80}")
    print(f"Processing with LOD: {lod}")
    print(f"{'='*80}\n")
    
    for dirpath, dirnames, filenames in os.walk(input):
        for filename in filenames:
            if filename.endswith('.geo'):
                input_geo_path = os.path.join(dirpath, filename).replace('\\', '/')
                relative_path = os.path.relpath(input_geo_path, input)
                basename = os.path.splitext(relative_path)[0].replace('\\', '_')
                try:
                    process_file(input_geo_path, basename, lod=lod)
                    print(f"✓ Successfully processed: {basename}\n")
                except Exception as e:
                    print(f"✗ Error processing {basename}: {str(e)}\n")