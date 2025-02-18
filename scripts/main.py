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

input = "D:/DATA/_cleaned/118/04504/04504-01.geo"


input_geo_path = rf"{user_profile}/AppData/Roaming/SketchUp/SketchUp 2022/SketchUp/Plugins/pkpm_moosas/data/geometry/selection0.geo"   

input_xml_path = rf"{user_profile}/AppData/Roaming/SketchUp/SketchUp 2022/SketchUp/Plugins/pkpm_moosas/data/geometry/selection0.xml" 

output_geo_path = "BuildingConvex/data/selection0_convex.geo"
output_xml_path = "BuildingConvex/data/selection0_convex.xml"
output_json_path = "BuildingConvex/data/adjson"

new_geo_path = "BuildingConvex/data/selection0_new.geo"
new_xml_path = "BuildingConvex/data/selection0_new.xml"

def convex_temp(input_geo_path, output_geo_path):
    cat, idd, normal, faces, holes = read_geo(input_geo_path)
    convex_cat, convex_idd, convex_normal, convex_faces = MoosasConvexify.convexify_faces(cat, idd, normal, faces, holes)
    write_geo (output_geo_path, convex_cat, convex_idd, convex_normal, convex_faces)
    


def graph_temp(new_geo_path, new_xml_path):
    graph = MoosasGraph()
    graph.graph_representation(new_geo_path, new_xml_path)  
    graph.draw_graph_3d()
    
    graph_to_json(graph, output_json_path)

convex_temp(input_geo_path, output_geo_path)
Moosas.transform(output_geo_path, new_xml_path, new_geo_path, divided_zones=False)
graph_temp(new_geo_path, new_xml_path)
