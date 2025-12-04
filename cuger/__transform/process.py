import os
from .convexify import MoosasConvexify
from .graph import MoosasGraph
from graphIO import read_geo, write_geo, graph_to_json


def get_output_paths(modelname, output_dir):
    """
    Generate output file paths for a given model.

    Args:
        modelname (str): The name of the model.
        output_dir (str): The base directory for output files.

    Returns:
        dict: A dictionary containing paths for various output files.
    """
    paths = {
        "output_geo_path": os.path.join(output_dir, "geo", f"{modelname}.geo"),
        "output_json_path": os.path.join(output_dir, "graph", f"{modelname}"),
        "new_xml_path": os.path.join(output_dir, "new_xml", f"{modelname}.xml"),
        "new_geo_path": os.path.join(output_dir, "new_geo", f"{modelname}.geo"),
        "new_idf_path": os.path.join(output_dir, "new_idf", f"{modelname}.idf"),
        "figure_convex_path": os.path.join(output_dir, "figure_convex", f"{modelname}_convex.png"),
        "figure_graph_path": os.path.join(output_dir, "figure_graph", f"{modelname}_graph.png"),
    }

    # Ensure all directories exist
    for path in paths.values():
        directory = os.path.dirname(path)
        if directory:  # Check if the directory part of the path is not empty
            os.makedirs(directory, exist_ok=True)

    return paths


def convex_process(input_geo_path, output_geo_path, figure_path=None):
    """
    Perform convexification on the input geometry file and save the result.

    Args:
        input_geo_path (str): Path to the input geometry file.
        output_geo_path (str): Path to save the convexified geometry file.
    """


    # Read geometry data
    cat, idd, normal, faces, holes = read_geo(input_geo_path)

    # Perform convexification
    convex_cat, convex_idd, convex_normal, convex_faces, divided_lines = MoosasConvexify.convexify_faces(
        cat, idd, normal, faces, holes
    )

    # Write convexified geometry data
    write_geo(output_geo_path, convex_cat, convex_idd, convex_normal, convex_faces)


    if figure_path:
        MoosasConvexify.plot_faces(convex_faces, divided_lines, file_path=figure_path)


def graph_process(new_geo_path, new_xml_path, output_json_path, figure_path=None):
    """
    Generate a graph representation from geometry and XML files, and save it as JSON.

    Args:
        new_geo_path (str): Path to the new geometry file.
        new_xml_path (str): Path to the new XML file.
        output_json_path (str): Path to save the graph as a JSON file.
    """

    # Initialize the graph
    graph = MoosasGraph()

    # Build graph representation
    graph.graph_representation(new_geo_path, new_xml_path)

    # Save the graph as JSON
    graph_to_json(graph, output_json_path)

    if figure_path:
        graph.draw_graph_3d(file_path=figure_path) 


