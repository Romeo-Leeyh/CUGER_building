import os, sys, time

from cuger.__transform.convexify import MoosasConvexify
from cuger.__transform.graph import MoosasGraph
import cuger.__transform.process as ps

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if main_dir not in sys.path:
    sys.path.append(main_dir)

import moosas.MoosasPy as Moosas
# moosas package is independent of cuger, so the import path is moosas.MoosasPy

input = "BuildingConvex/results/examples"
output = "BuildingConvex/results/example_results"

def process_file(input_geo_path, modelname):
    paths = ps.get_output_paths(modelname, output)

    print(f"Processing file: {input_geo_path}, basename: {modelname}")
    

    try:
        ps.convex_process(input_geo_path, paths["output_geo_path"], paths["figure_convex_path"])

        model = Moosas.transform(input_geo_path, paths["new_xml_path"], paths["new_geo_path"], 
                        solve_overlap=True, 
                        divided_zones=False, 
                        break_wall_horizontal=True, 
                        solve_redundant=True,
                        attach_shading=False,
                        standardize=True)

        Moosas.saveModel(model, paths["new_rdf_path"])
        
        ps.graph_process(paths["new_geo_path"], paths["new_xml_path"], paths["output_json_path"], paths["figure_graph_path"])
    except ValueError as e:
        print(f"ValueError: {e} - Modelname: {modelname}")
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e} - Modelname: {modelname}")
    except Exception as e:
        print(f"Unexpected error: {e} - Modelname: {modelname}")
    

for dirpath, dirnames, filenames in os.walk(input):
    for filename in filenames:
        if filename.endswith('.geo'):
            
            input_geo_path = os.path.join(dirpath, filename).replace('\\', '/')
            relative_path = os.path.relpath(input_geo_path, input)
            basename = os.path.splitext(relative_path)[0].replace('\\', '_')
            process_file(input_geo_path, basename)