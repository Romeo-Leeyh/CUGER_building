import os
import sys

from convexify import MoosasConvexify
from graph import MoosasGraph
from graphIO import read_geo, write_geo, read_xml, graph_to_json


main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if main_dir not in sys.path:
    sys.path.append(main_dir)

import moosas.python.Lib.MoosasPy as Moosas

#main
user_profile = os.environ['USERPROFILE']
input = rf"{user_profile}/AppData/Roaming/SketchUp/SketchUp 2022/SketchUp/Plugins/pkpm_moosas/data/geometry/"
output = "BuildingConvex/data"
figure_path = "BuildingConvex/data/figure/"


def get_output_paths(modelname):
    return {
        "output_geo_path": os.path.join(output, "geo", f"{modelname}.geo"),
        "output_json_path": os.path.join(output, "graph", f"{modelname}"),
        "new_xml_path": os.path.join(output, "new_xml", f"{modelname}.xml"),
        "new_geo_path": os.path.join(output, "new_geo", f"{modelname}.geo"),
        "figure_path": os.path.join(output, "figure", f"{modelname}.png")
    }

def convex_temp(input_geo_path, output_geo_path):
    cat, idd, normal, faces, holes = read_geo(input_geo_path)
    convex_cat, convex_idd, convex_normal, convex_faces, divided_lines = MoosasConvexify.convexify_faces(cat, idd, normal, faces, holes)
    write_geo(output_geo_path, convex_cat, convex_idd, convex_normal, convex_faces)
    MoosasConvexify.plot_faces(convex_faces, divided_lines, file_path=figure_path)


def graph_temp(new_geo_path, new_xml_path, output_json_path):
    graph = MoosasGraph()
    graph.graph_representation(new_geo_path, new_xml_path) 
    graph.draw_graph_3d(file_path=figure_path) 
    graph_to_json(graph, output_json_path)

def process_file(input_geo_path, modelname):
    paths = get_output_paths(modelname)
    
    """
    if os.path.exists(paths["output_geo_path"]):
        print(f"--Skip-- | {modelname}")
        #return
    """
    print(f"Processing file: {input_geo_path}, basename: {modelname}")
    
    #convex_temp(input_geo_path, paths["output_geo_path"])
    #Moosas.transform(paths["output_geo_path"], paths["new_xml_path"], paths["new_geo_path"], divided_zones=False,  standardize=True)
    graph_temp(paths["new_geo_path"], paths["new_xml_path"], paths["output_json_path"])
     
def process_geo_files(input_dir):
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.geo'):
                
                input_geo_path = os.path.join(dirpath, filename).replace('\\', '/')
                print (f"Processing {input_geo_path}")

                basename = filename.split('.')[0]
                process_file(input_geo_path, basename)
                break

process_geo_files(input)

