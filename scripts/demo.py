import os

from convexify import MoosasConvexify
from graph import MoosasGraph
from graphIO import read_geo, write_geo, read_xml, graph_to_json
import sys

main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if main_dir not in sys.path:
    sys.path.append(main_dir)

import moosas.python.Lib.MoosasPy as Moosas

#main
user_profile = os.environ['USERPROFILE']

input = "E:/DATA/Moosasbuildingdatasets/_cleaned"
output = "E:/DATA/Moosasbuildingdatasets/"


def get_output_paths(modelname):
    return {
        "output_geo_path": os.path.join(output, "geo", f"{modelname}.geo"),
        "output_json_path": os.path.join(output, "graph", f"{modelname}"),
        "new_xml_path": os.path.join(output, "new_xml", f"{modelname}.xml"),
        "new_geo_path": os.path.join(output, "new_geo", f"{modelname}.geo"),
        "figure_path": os.path.join(output, "figure", f"convex/{modelname}.png"),
        "figure0_path": os.path.join(output, "figure", f"original/{modelname}.png"),
        "figure_graph_path": os.path.join(output, "figure", f"graph/{modelname}.png")
    }

def convex_temp(input_geo_path, output_geo_path, figure0_path, figure_path):
    cat, idd, normal, faces, holes = read_geo(input_geo_path)
    MoosasConvexify.plot_faces(faces, None, file_path=figure0_path)
    convex_cat, convex_idd, convex_normal, convex_faces, divided_lines = MoosasConvexify.convexify_faces(cat, idd, normal, faces, holes)
    write_geo(output_geo_path, convex_cat, convex_idd, convex_normal, convex_faces)
    MoosasConvexify.plot_faces(convex_faces, divided_lines, file_path=figure_path)
    F, E, V = MoosasConvexify.calculate(convex_faces)
    print(f"Number of faces: {F}, Number of edges: {E}, Number of vertices: {V}, Euler number: {V - E + F}")

def graph_temp(new_geo_path, new_xml_path, output_json_path, figure_graph_path):
    graph = MoosasGraph()
    graph.graph_representation(new_geo_path, new_xml_path) 
    graph.draw_graph_3d(figure_graph_path) 
    graph_to_json(graph, output_json_path)

def process_file(input_geo_path, modelname):
    paths = get_output_paths(modelname)

    
    print(f"Processing file: {input_geo_path}, basename: {modelname}")
    
    """
    try:
        convex_temp(input_geo_path, paths["output_geo_path"])
        Moosas.transform(paths["output_geo_path"], paths["new_xml_path"], paths["new_geo_path"], divided_zones=False, standardize=True)
        graph_temp(paths["new_geo_path"], paths["new_xml_path"], paths["output_json_path"])
    except ValueError as e:
        print(f"ValueError: {e} - Modelname: {modelname}")
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e} - Modelname: {modelname}")
    except Exception as e:
        print(f"Unexpected error: {e} - Modelname: {modelname}")
    
    """
    #convex_temp(input_geo_path, paths["output_geo_path"], paths["figure0_path"], paths["figure_path"])
    #Moosas.transform(paths["output_geo_path"], paths["new_xml_path"], paths["new_geo_path"], divided_zones=False, stdout=None, standardize=True)
    graph_temp(paths["new_geo_path"], paths["new_xml_path"], paths["output_json_path"], paths["figure_graph_path"])
     
    
    
# 遍历文件夹，处理所有 .geo 文件
def process_geo_files(input_dir):
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.geo'):
                
                input_geo_path = os.path.join(dirpath, filename).replace('\\', '/')
                relative_path = os.path.relpath(input_geo_path, input_dir)
                basename = os.path.splitext(relative_path)[0].replace('\\', '_')
                process_file(input_geo_path, basename)
                


# 执行处理

process_geo_files(input)

