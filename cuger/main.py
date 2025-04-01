import os

from __transform.convexify import MoosasConvexify
from __transform.graph import MoosasGraph
import __transform.process as ps
import sys

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if main_dir not in sys.path:
    sys.path.append(main_dir)

import moosas.python.Lib.MoosasPy as Moosas

#main
user_profile = os.environ['USERPROFILE']

input = "E:/DATA/Moosasbuildingdatasets_02/_cleaned"
output = "E:/DATA/Moosasbuildingdatasets_02/output_0"
_fig_show = False

def process_file(input_geo_path, modelname):
    paths = ps.get_output_paths(modelname, output)
    if os.path.exists(paths["output_geo_path"]):
        print(f"--Skip-- | {modelname}")
        return
    print(f"Processing file: {input_geo_path}, basename: {modelname}")
    
    
    try:
        ps.convex_process(input_geo_path, paths["output_geo_path"], paths["figure_convex_path"])
        Moosas.transform(input_geo_path, paths["new_xml_path"], paths["new_geo_path"], divided_zones=False, standardize=True)
        ps.graph_process(paths["new_geo_path"], paths["new_xml_path"], paths["output_json_path"], paths["figure_graph_path"])
    except ValueError as e:
        print(f"ValueError: {e} - Modelname: {modelname}")
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e} - Modelname: {modelname}")
    except Exception as e:
        print(f"Unexpected error: {e} - Modelname: {modelname}")
    
    """
    convex_temp(input_geo_path, paths["output_geo_path"])
    Moosas.transform(paths["output_geo_path"], paths["new_xml_path"], paths["new_geo_path"], divided_zones=False,  standardize=True)
    graph_temp(paths["new_geo_path"], paths["new_xml_path"], paths["output_json_path"])"
    """
     


# 执行处理
input = rf"{user_profile}/AppData/Roaming/SketchUp/SketchUp 2022/SketchUp/Plugins/pkpm_moosas/data/geometry/"

for dirpath, dirnames, filenames in os.walk(input):
    for filename in filenames:
        if filename.endswith('.geo'):
            
            input_geo_path = os.path.join(dirpath, filename).replace('\\', '/')
            relative_path = os.path.relpath(input_geo_path, input)
            basename = os.path.splitext(relative_path)[0].replace('\\', '_')
            process_file(input_geo_path, basename)

