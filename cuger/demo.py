import os
import sys

from __transform.convexify import MoosasConvexify
from __transform.graph import MoosasGraph
from graphIO import read_geo, write_geo, read_xml, graph_to_json


main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if main_dir not in sys.path:
    sys.path.append(main_dir)

import moosas.python.Lib.MoosasPy as Moosas

#main
user_profile = os.environ['USERPROFILE']
input = rf"{user_profile}/AppData/Roaming/SketchUp/SketchUp 2022/SketchUp/Plugins/pkpm_moosas/data/geometry/"
input = "BuildingConvex\data\sample"
input = "E:/DATA/CUGER_buildingdatasets/1028_17_46_89_255"
output = "E:/DATA/CUGER_buildingdatasets/results"
#input = "E:/DATA/Daylighting_test/model/evomass/geo"
#output = "BuildingConvex/data"
_fig_show = True

def get_output_paths(modelname):
    return {
        "output_geo_path": os.path.join(output, "geo", f"{modelname}.geo"),
        "output_json_path": os.path.join(output, "graph", f"{modelname}"),
        "new_xml_path": os.path.join(output, "new_xml", f"{modelname}.xml"),
        "new_geo_path": os.path.join(output, "new_geo", f"{modelname}.geo"),
        "figure_path": os.path.join(output, "figure", f"{modelname}.png"),
        "figure_convex_path": os.path.join(output, "figure_convex", f"{modelname}_convex.png"),
        "figure_graph_path": os.path.join(output, "figure_graph", f"{modelname}_graph.png"),
    }

def convex_temp(input_geo_path, output_geo_path, figure_path):
    cat, idd, normal, faces, holes = read_geo(input_geo_path)
    print (f"Read {len(faces)} faces from {input_geo_path}")
    convex_cat, convex_idd, convex_normal, convex_faces, divided_lines = MoosasConvexify.convexify_faces(cat, idd, normal, faces, holes)

    write_geo(output_geo_path, convex_cat, convex_idd, convex_normal, convex_faces)
    MoosasConvexify.plot_faces(convex_faces, divided_lines, file_path=figure_path, _fig_show=_fig_show)


def graph_temp(new_geo_path, new_xml_path, output_json_path, figure_path):
    graph = MoosasGraph()
    graph.graph_representation(new_geo_path, new_xml_path) 
    graph.draw_graph_3d(file_path=figure_path, _fig_show=_fig_show) 
    graph_to_json(graph, output_json_path)

def process_file(input_geo_path, modelname):
    paths = get_output_paths(modelname)
    
    """
    if os.path.exists(paths["output_geo_path"]):
        print(f"--Skip-- | {modelname}")
        #return
    """
    print(f"Processing file: {input_geo_path}, basename: {modelname}")
    
    convex_temp(input_geo_path, paths["output_geo_path"], paths["figure_convex_path"])
    Moosas.transform(input_geo_path, paths["new_xml_path"], paths["new_geo_path"], 
                     solve_contains=False, 
                     divided_zones=True, 
                     break_wall_horizontal=True, 
                     solve_redundant=True,
                     attach_shading=False,
                     standardize=True)
    graph_temp(paths["new_geo_path"], paths["new_xml_path"], paths["output_json_path"], paths["figure_graph_path"])
     
def process_geo_files(input_dir):
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            #if filename.endswith('20250416_143152.geo'):
            #if filename.endswith('test.geo'):  
             
            #if filename.endswith('20250416_143152.geo'):      
            if filename.endswith('.geo'):    
                input_geo_path = os.path.join(dirpath, filename).replace('\\', '/')
                print (f"Processing {input_geo_path}")

                basename = filename.split('.')[0]
                process_file(input_geo_path, basename)
                

process_geo_files(input)

