import os

from convexify import MoosasConvexify
from graph import MoosasGraph
from graphIO import read_geo, write_geo, read_xml, graph_to_json
import sys

input = "C:/Users/LI YIHUI/Downloads/selection0"

def convex_temp(input_geo_path):
    cat, idd, normal, faces, holes = read_geo(input_geo_path)
    convex_cat, convex_idd, convex_normal, convex_faces, divided_lines = MoosasConvexify.convexify_faces(cat, idd, normal, faces, holes)
    MoosasConvexify.plot_faces(convex_faces, divided_lines)
    F, E, V = MoosasConvexify.calculate(convex_faces)
    print(f"Number of faces: {F}, Number of edges: {E}, Number of vertices: {V}, Euler number: {V - E + F}")


for dirpath, dirnames, filenames in os.walk(input):
    for filename in filenames:
        if filename.endswith('.geo'):
            input_geo_path = os.path.join(dirpath, filename).replace('\\', '/')
            convex_temp(input_geo_path)