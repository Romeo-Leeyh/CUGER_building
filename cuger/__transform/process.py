from pathlib import Path
from .convexify import *
from .simplify import simplify_faces, inject_minimal_core
from .graph import MoosasGraph
from ..graphIO import *
from ..__analyse.visualise import plot_convex_faces, plot_graph_3d


def get_output_paths(modelname, output_dir, lod="precise"):
    """
    Generate output file paths for a given model.

    Args:
        modelname (str): The name of the model.
        output_dir (str): The base directory for output files.

    Returns:
        dict: A dictionary containing paths for various output files.
    """
    output_root = Path(output_dir)
    paths = {
        "simplified_geo_path": output_root / "geo_s" / f"{modelname}_s_{lod}.geo",
        "convex_geo_path": output_root / "geo_c" / f"{modelname}_c.geo",
        "output_graph_path": output_root / "graph" / f"{modelname}.json",
        "new_xml_path": output_root / "new_xml" / f"{modelname}.xml",
        "new_geo_path": output_root / "new_geo" / f"{modelname}.geo",
        "new_idf_path": output_root / "new_idf" / f"{modelname}.idf",
        "new_rdf_path": output_root / "new_rdf" / f"{modelname}.owl",
        "figure_convex_path": output_root / "figure_convex" / f"{modelname}_convex.png",
        "figure_graph_path": output_root / "figure_graph" / f"{modelname}_graph.png",
    }

    # Ensure all directories exist
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    return {key: str(value) for key, value in paths.items()}


def simplify_process(
    input_geo_path,
    output_geo_path,
    lod="precise",
    enable_minimal_core=False,
):
    """
    Simplify the geometry in the input geo file and save the result.

    Args:
        input_geo_path (str): Path to the input geometry file.
        output_geo_path (str): Path to save the simplified geometry file.
        figure_path (str, optional): Path to save a figure of the simplified geometry. Defaults to None.
        lod (str, optional): Level of detail for simplification ("precise", "medium", "low"). Defaults to "precise".
        enable_minimal_core (bool, optional): If True, inject a minimal
            core shaft into low/medium simplified geometry before writing
            the simplified output. Defaults to False.
    """
    # Read geometry data
    cat, idd, normal, faces, holes = read_geo(input_geo_path)

    output_geo = Path(output_geo_path)
    output_geo.parent.mkdir(parents=True, exist_ok=True)

    # Perform simplification
    if lod == "precise":
        # Copy the input geo file to the output path without modification
        content = Path(input_geo_path).read_text(encoding="utf-8")
        output_geo.write_text(content, encoding="utf-8")
        return
    elif lod in ["medium", "low"]:
        simplified_cat, simplified_idd, simplified_normal, simplified_faces, simplified_holes = simplify_faces(
            cat, idd, normal, faces, holes, lod=lod
        )
    else:
        raise ValueError("lod must be one of: precise, medium, low")

    if enable_minimal_core and lod in ["medium", "low"]:
        simplified_cat, simplified_idd, simplified_normal, simplified_faces, simplified_holes = inject_minimal_core(
            simplified_cat,
            simplified_idd,
            simplified_normal,
            simplified_faces,
            simplified_holes,
            lod=lod,
        )

    # Write simplified geometry data
    write_geo(output_geo_path, simplified_cat, simplified_idd, simplified_normal, simplified_faces, simplified_holes)


def convex_process(input_geo_path, output_geo_path, figure_path=None, overlay_geo_path=None):
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
    Path(output_geo_path).parent.mkdir(parents=True, exist_ok=True)
    write_geo(output_geo_path, convex_cat, convex_idd, convex_normal, convex_faces)


    if figure_path:
        overlay_faces = None
        if overlay_geo_path:
            _, _, _, overlay_faces, _ = read_geo(overlay_geo_path)
        plot_convex_faces(convex_faces, divided_lines, file_path=figure_path, overlay_faces=overlay_faces)


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

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize the graph
    graph = MoosasGraph()

    # Build graph representation
    graph.graph_representation_new(root, faces_category, faces_id, faces_normal, faces_vertices, faces_holes)

    graph.graph_edit(_isolated_clean=True, _airwall_clean=True)

    # Save the graph as JSON
    graph_to_json(graph, output_json_path)

    if figure_path:
        plot_graph_3d(graph.graph, file_path=figure_path) 


