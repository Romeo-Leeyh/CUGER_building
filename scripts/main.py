import os

from convexify import MoosasConvexify
from graph import MoosasGraph
from graphIO import read_geo, write_geo, read_xml, graph_to_json


#main
user_profile = os.environ['USERPROFILE']

input_geo_path = rf"{user_profile}/AppData/Roaming/SketchUp/SketchUp 2022/SketchUp/Plugins/pkpm_moosas/data/geometry/selection0_out.geo"   

input_xml_path = rf"{user_profile}/AppData/Roaming/SketchUp/SketchUp 2022/SketchUp/Plugins/pkpm_moosas/data/geometry/selection0.xml" 

output_geo_path = "data/selection0_convex.geo"
output_xml_path = "data/selection0_convex.xml"
output_json_path = "data/adjson"


def convex_temp():
    cat, idd, normal, faces, holes = read_geo(input_geo_path)
    convex_cat, convex_idd, convex_normal, convex_faces = MoosasConvexify.convexify_faces(cat, idd, normal, faces, holes)
    write_geo (output_geo_path, convex_cat, convex_idd, convex_normal, convex_faces)
    


def graph_temp():
    graph = MoosasGraph()
    graph.graph_representation(input_geo_path, input_xml_path)  
    graph.draw_graph_3d()
    
    graph_to_json(graph, output_json_path)

#convex_temp()
graph_temp()
