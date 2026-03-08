import os
from .convexify import *
from .simplify import simplify_faces
from .graph import MoosasGraph
from graphIO import *

# Import visualization functions
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from __analyse.visualise import plot_convex_faces, plot_graph_3d


def get_output_paths(modelname, output_dir, lod="precise"):
    """
    Generate output file paths for a given model.

    Args:
        modelname (str): The name of the model.
        output_dir (str): The base directory for output files.

    Returns:
        dict: A dictionary containing paths for various output files.
    """
    paths = {
        "simplified_geo_path": os.path.join(output_dir, "geo_s", f"{modelname}_s_{lod}.geo"),
        "convex_geo_path": os.path.join(output_dir, "geo_c", f"{modelname}_c.geo"),
        "output_graph_path": os.path.join(output_dir, "graph", f"{modelname}.json"),
        "new_xml_path": os.path.join(output_dir, "new_xml", f"{modelname}.xml"),
        "new_geo_path": os.path.join(output_dir, "new_geo", f"{modelname}.geo"),
        "new_idf_path": os.path.join(output_dir, "new_idf", f"{modelname}.idf"),
        "new_rdf_path": os.path.join(output_dir, "new_rdf", f"{modelname}.owl"),
        "figure_convex_path": os.path.join(output_dir, "figure_convex", f"{modelname}_convex.png"),
        "figure_graph_path": os.path.join(output_dir, "figure_graph", f"{modelname}_graph.png"),
    }

    # Ensure all directories exist
    for path in paths.values():
        directory = os.path.dirname(path)
        if directory:  # Check if the directory part of the path is not empty
            os.makedirs(directory, exist_ok=True)

    return paths


def simplify_process(input_geo_path, output_geo_path, figure_path=None, lod="precise"):
    """
    Simplify the geometry in the input geo file and save the result.

    Args:
        input_geo_path (str): Path to the input geometry file.
        output_geo_path (str): Path to save the simplified geometry file.
        figure_path (str, optional): Path to save a figure of the simplified geometry. Defaults to None.
        lod (str, optional): Level of detail for simplification ("precise", "medium", "low"). Defaults to "precise".
    """
    # Read geometry data
    cat, idd, normal, faces, holes = read_geo(input_geo_path)

    # Perform simplification
    if lod == "precise":
        # Copy the inoput geo file to the output path without modification
        with open(input_geo_path, "r") as f:
            content = f.read()
        with open(output_geo_path, "w") as f:
            f.write(content)
        return
    elif lod in ["medium", "low"]:
        simplified_cat, simplified_idd, simplified_normal, simplified_faces, simplified_holes = simplify_faces(
            cat, idd, normal, faces, holes, lod=lod
        )

    # Write simplified geometry data
    write_geo(output_geo_path, simplified_cat, simplified_idd, simplified_normal, simplified_faces, simplified_holes)


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
    convex_cat, convex_idd, convex_normal, convex_faces, divided_lines = convexify_faces(
        cat, idd, normal, faces, holes
    )

    # Write convexified geometry data
    write_geo(output_geo_path, convex_cat, convex_idd, convex_normal, convex_faces)


    if figure_path:
        plot_convex_faces(convex_faces, divided_lines, file_path=figure_path)


def graph_process(new_geo_path, new_xml_path, output_json_path, figure_path=None):
    """
    Generate a graph representation from geometry and XML files, and save it as JSON.

    Args:
        new_geo_path (str): Path to the new geometry file.
        new_xml_path (str): Path to the new XML file.
        output_json_path (str): Path to save the graph as a JSON file.
    """
    faces_category, faces_id, faces_normal, faces_vertices, faces_holes = read_geo(new_geo_path)
    root = read_xml(new_xml_path)
    
    # Initialize the graph
    graph = MoosasGraph()

    # Build graph representation
    graph.graph_representation_new(root, faces_category, faces_id, faces_normal, faces_vertices, faces_holes)

    graph.graph_edit(_isolated_clean=True, _airwall_clean=True)

    # Save the graph as JSON
    graph_to_json(graph, output_json_path)

    if figure_path:
        plot_graph_3d(graph.graph, file_path=figure_path) 


