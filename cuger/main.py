import os, sys, time

from __transform.convexify import MoosasConvexify
from __transform.graph import MoosasGraph
import __transform.process as ps

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if main_dir not in sys.path:
    sys.path.append(main_dir)

import moosas.MoosasPy as Moosas

#main
user_profile = os.environ['USERPROFILE']

#input = "E:/DATA/CUGER_buildingdatasets/1028_17_46_89_255"
#output = "E:/DATA/CUGER_buildingdatasets/results"
_fig_show = True
input = "results/example"
output = "results/example_results"


def process_file(input_geo_path, modelname):
    paths = ps.get_output_paths(modelname, output)
    if os.path.exists(paths["new_xml_path"]):
        print(f"--Skip-- | {modelname}")
        
    print(f"Processing file: {input_geo_path}, basename: {modelname}")
    
    """
    try:
        ps.convex_process(input_geo_path, paths["output_geo_path"], paths["figure_convex_path"])

        Moosas.transform(input_geo_path, paths["new_xml_path"], paths["new_geo_path"], 
                        solve_contains=False, 
                        divided_zones=True, 
                        break_wall_horizontal=True, 
                        solve_redundant=True,
                        attach_shading=False,
                        standardize=True)

        ps.graph_process(paths["new_geo_path"], paths["new_xml_path"], paths["output_json_path"], paths["figure_graph_path"])
    except ValueError as e:
        print(f"ValueError: {e} - Modelname: {modelname}")
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e} - Modelname: {modelname}")
    except Exception as e:
        print(f"Unexpected error: {e} - Modelname: {modelname}")
    
    """
    ps.convex_process(input_geo_path, paths["output_geo_path"], paths["figure_convex_path"])
    Moosas.transform(paths, 
                    solve_contains=False, 
                    divided_zones=True, 
                    break_wall_horizontal=True, 
                    solve_redundant=True,
                    attach_shading=False,
                    standardize=True)

    ps.graph_process(paths["new_geo_path"], paths["new_xml_path"], paths["output_json_path"], paths["figure_graph_path"])
    
  



# 执行处理
#input = rf"{user_profile}/AppData/Roaming/SketchUp/SketchUp 2022/SketchUp/Plugins/pkpm_moosas/data/geometry/"

for dirpath, dirnames, filenames in os.walk(input):
    for filename in filenames:
        if filename.endswith('.geo'):
            
            input_geo_path = os.path.join(dirpath, filename).replace('\\', '/')
            relative_path = os.path.relpath(input_geo_path, input)
            basename = os.path.splitext(relative_path)[0].replace('\\', '_')
            process_file(input_geo_path, basename)

