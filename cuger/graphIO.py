"""
    This is the i/o module of encodling and representation algorithm. Different from the Moosas+ i/o module, this algorithm do not use pygeos to process geometry !!!
"""

import numpy as np
import xml.etree.ElementTree as ET
import json
import networkx as nx
from pathlib import Path

def read_geo(file_path):
    """
    Reads a .geo file and returns categories, IDs, normal vectors, and polygon face coordinates.

    Parameters:
        file_path (str): File path of the .geo file.

    Returns:
        tuple: including (cat, idd, normal, faces, holes)
    """
    cat, idd, normal, faces, holes = [], [], [], [], []

    def next_nonempty_line(file):

        line = file.readline()
        while line and line.strip() == '':
            line = file.readline()
        return line.strip()

    try:
        with open(file_path, "r", encoding='utf-8') as f:
            read = next_nonempty_line(f)
            
            while read:
                if read.startswith("f,"):
                    parts = read.split(',')
                    cat.append(str(parts[1]))
                    idd.append(str(parts[2]))
                    read = next_nonempty_line(f)
                    continue

                if read.startswith("fn,"):
                    normal_parts = read.split(',')
                    normal_t = np.array([float(normal_parts[1]), float(normal_parts[2]), float(normal_parts[3])])
                    normal.append(normal_t)
                    read = next_nonempty_line(f)
                    continue

                if read.startswith("fv,"):
                    vertices = []
                    while read.startswith("fv,"):
                        vertex_parts = read.split(',')
                        vertex = np.array([float(vertex_parts[1]), float(vertex_parts[2]), float(vertex_parts[3])])
                        vertices.append(vertex)
                        read = next_nonempty_line(f)
                    faces.append(np.array(vertices))
                    holes.append(None)
                    continue

                if read.startswith("fh,"):
                    current_face_holes = {}
                    while read.startswith("fh,"):
                        vertex_parts = read.split(',')
                        hole_id = int(vertex_parts[1])
                        vertex = np.array([float(vertex_parts[2]), float(vertex_parts[3]), float(vertex_parts[4])])
                        if hole_id not in current_face_holes:
                            current_face_holes[hole_id] = []
                        current_face_holes[hole_id].append(vertex)
                        read = next_nonempty_line(f)
                    holes[-1] = current_face_holes.copy()
                    continue

                read = next_nonempty_line(f)



    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    
    for face_holes in holes:
        if face_holes is not None:
            for hole_id in face_holes:
                face_holes[hole_id] = np.array(face_holes[hole_id])
    
    # Return the parsed data
    return np.array(cat), idd, np.array(normal), faces, holes

def write_geo(file_path, cat, idd, normal, faces):
    """
    Write categories, IDs, normal vectors, and polygon face information to a .geo file.

    Parameters:
        output_file_path (str): File path of the output .geo file.
        cat (list): List of categories.
        idd (list): List of IDs.
        normal (list): List of normal vectors.
        faces (list): List of polygon face information.
    """
    try:
        with open(file_path, "w", encoding='utf-8') as f:
            for i in range(len(cat)):
                # Write face category and ID
                f.write(f"f,{cat[i]},{idd[i]}\n")
                
                # Write normal vector
                normal_vector = normal[i]
                f.write(f"fn,{normal_vector[0]},{normal_vector[1]},{normal_vector[2]}\n")
                
                # Write face vertices
                for vertex in faces[i]:
                    f.write(f"fv,{vertex[0]},{vertex[1]},{vertex[2]}\n")
                
                
                # Write a separator if it's not the last face
                if i != len(cat) - 1:
                    f.write(";\n")
    
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")

def read_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return root

def write_adjson (file_path, data):
    try:
        with open(file_path, "w", encoding='utf-8') as f:
            f.write(data)
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")

def read_adjson (file_path):
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            data = f.read()
            return data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

class NumpyEncoder(json.JSONEncoder):
    """
        JSON encoder that handles numpy arrays
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return super().default(obj)

def graph_to_json(graph, output_dir):
    """
        Dumps graph data to JSON files
    """
    # Prepare node data
    nodes_data = {}
    for node_id, node_attrs in graph.graph.nodes(data=True):
        node_data = {}
        for key, value in node_attrs.items():
            if isinstance(value, dict):
                node_data[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                 for k, v in value.items()}
            else:
                node_data[key] = value.tolist() if isinstance(value, np.ndarray) else value
        nodes_data[str(node_id)] = node_data
    
    # Prepare edge data
    edges_data = {}
    for u, v, edge_attrs in graph.graph.edges(data=True):
        edge_id = f"{u}_{v}"
        edges_data[edge_id] = {
            "source": str(u),
            "target": str(v),
            "attributes": edge_attrs
        }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write to JSON files
    with open(output_path / "nodes.json", "w", encoding="utf-8") as f:
        json.dump(nodes_data, f, indent=4, cls=NumpyEncoder)
    
    with open(output_path / "edges.json", "w", encoding="utf-8") as f:
        json.dump(edges_data, f, indent=4, cls=NumpyEncoder)

def json_to_graph(input_dir):
    """Reconstructs graph data from JSON files.
    
    Parameters:
        input_dir (str): JSON files directory.
        
    Returns:
        nx.Graph: Reconstructed NetworkX graph.
    """
    input_path = Path(input_dir)
    
    # Read node data
    with open(input_path / "nodes.json", "r", encoding="utf-8") as f:
        nodes_data = json.load(f)
    
    # Read edge data
    with open(input_path / "edges.json", "r", encoding="utf-8") as f:
        edges_data = json.load(f)
    
    # Construct the graph
    G = nx.Graph()
    
    # Add nodes and attributes
    for node_id, node_attrs in nodes_data.items():
        
        processed_attrs = {}
        for key, value in node_attrs.items():
            if isinstance(value, dict):
                processed_attrs[key] = {k: np.array(v) if isinstance(v, list) else v 
                                     for k, v in value.items()}
            else:
                processed_attrs[key] = np.array(value) if isinstance(value, list) else value
        G.add_node(node_id, **processed_attrs)
    
    # Add edges and attributes
    for edge_id, edge_data in edges_data.items():
        source = edge_data["source"]
        target = edge_data["target"]
        attributes = edge_data["attributes"]
        G.add_edge(source, target, **attributes)
    
    return G



