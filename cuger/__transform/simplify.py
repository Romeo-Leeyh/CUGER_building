"""
    This module provides functions to simplify a geo file to different levels of detail. The main function is `simplify_faces`, which takes a batch of faces and produces a simplified version based on the specified level of detail (LOD). The LOD can be set to "precise", "medium", or "low", which determines the degree of simplification applied to the geometry.
    
    - precise: output original geometry without simplification
    - medium: output multi-layer OBB extraction based on embedding layers
    - low: output the entire OBB as a single simplified geometry
"""

import numpy as np
from .geometry import create_obb, obb_to_face_vertices


def simplify_faces(cat, idd, normal, faces, holes, lod="precise"):
    """
    Simplify polygonal faces to different levels of detail.
    
    Parameters
    ----------
    cat : list[str]
        Category ID of each face.
    idd : list[str]
        Identifier of each face.
    normal : list[array-like]
        Normal vectors of faces.
    faces : list[list[array-like]]
        Vertex sequences of each face (outer boundary).
    holes : list[list[list[array-like]]]
        Hole vertex sequences for each face (may be empty).
    lod : str, optional
        Level of detail: "precise", "medium", or "low". Defaults to "precise".
        - "precise": return original faces without simplification
        - "medium": return multi-layer OBB extraction
        - "low": return entire OBB as single simplified geometry
    
    Returns
    -------
    tuple
        (simplified_cat, simplified_idd, simplified_normal, simplified_faces)
    """
    
    if lod == "precise":
        # Return original geometry without simplification
        return cat, idd, normal, faces, holes
    
    elif lod == "low":
        # Return entire OBB as single simplified geometry
        return _simplify_to_single_obb(cat, idd, normal, faces)
    
    elif lod == "medium":
        # Return multi-layer OBB extraction
        return _simplify_to_multi_layer_obb(cat, idd, normal, faces)
    
    else:
        # Default if unknown LOD
        return cat, idd, normal, faces, holes


def _simplify_to_single_obb(cat, idd, normal, faces):
    """
    Simplify geometry to a single OBB (low LOD).
    
    Combines all face vertices into one OBB and returns 6 faces representing the box.
    
    Parameters
    ----------
    cat : list[str]
        Category IDs.
    idd : list[str]
        Identifiers.
    normal : list[array-like]
        Normal vectors.
    faces : list[list[array-like]]
        Face vertices.
    
    Returns
    -------
    tuple
        (simplified_cat, simplified_idd, simplified_normal, simplified_faces)
    """
    # Combine all vertices into single point cloud
    all_verts = np.vstack(faces)
    
    # Calculate average normal (or use z-axis if all normals are similar)
    avg_normal = np.mean(normal, axis=0)
    avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-10)
    
    # Create OBB for all points
    obb_params = create_obb(all_verts, avg_normal)
    obb_faces = obb_to_face_vertices(obb_params)
    
    # Generate category (all faces type=1), ID and normal for OBB faces
    simplified_cat = [0] * 6  # Face type is 1
    simplified_idd = [f'obb_face_{i}' for i in range(6)]
    simplified_normal = []
    
    for face_verts in obb_faces:
        # Calculate normal from vertices
        v1 = face_verts[1] - face_verts[0]
        v2 = face_verts[3] - face_verts[0]
        face_normal = np.cross(v1, v2)
        face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-10)
        simplified_normal.append(face_normal)
    
    return (
        np.array(simplified_cat),
        simplified_idd,
        np.array(simplified_normal),
        obb_faces
    )


def _simplify_to_multi_layer_obb(cat, idd, normal, faces):
    """
    Simplify geometry to multi-layer OBB extraction (medium LOD).
    
    Extracts outer layers and generates OBB for each layer.
    
    Parameters
    ----------
    cat : list[str]
        Category IDs.
    idd : list[str]
        Identifiers.
    normal : list[array-like]
        Normal vectors.
    faces : list[list[array-like]]
        Face vertices.
    
    Returns
    -------
    tuple
        (simplified_cat, simplified_idd, simplified_normal, simplified_faces)
    """
    # For medium LOD, we use a simplified approach:
    # Create OBBs for different groups of faces (e.g., outer faces and inner faces)
    
    simplified_cat = []
    simplified_idd = []
    simplified_normal = []
    simplified_faces_list = []
    
    if len(faces) == 0:
        return (np.array([]), [], np.array([]), [])
    
    # Group faces by their normal direction to create layers
    # Separate faces into roof/floor and walls based on normal
    roof_floor_faces = []
    wall_faces = []
    roof_floor_normals = []
    wall_normals = []
    
    for i, face_normal in enumerate(normal):
        # Check if normal is mostly vertical (roof/floor) or horizontal (wall)
        if abs(face_normal[2]) > 0.7:  # Mostly vertical
            roof_floor_faces.append(faces[i])
            roof_floor_normals.append(face_normal)
        else:  # Mostly horizontal
            wall_faces.append(faces[i])
            wall_normals.append(face_normal)
    
    # Generate OBB for roof/floor faces (Layer 1)
    if len(roof_floor_faces) > 0:
        layer_faces = roof_floor_faces
        layer_normals = roof_floor_normals
        all_verts = np.vstack(layer_faces)
        avg_normal = np.mean(layer_normals, axis=0)
        avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-10)
        
        obb_params = create_obb(all_verts, avg_normal)
        obb_faces = obb_to_face_vertices(obb_params)
        
        for face_idx, face_verts in enumerate(obb_faces):
            simplified_cat.append(1)  # Face type is 1
            simplified_idd.append(f'layer1_obb_face_{face_idx}')
            
            v1 = face_verts[1] - face_verts[0]
            v2 = face_verts[3] - face_verts[0]
            face_normal = np.cross(v1, v2)
            face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-10)
            simplified_normal.append(face_normal)
            simplified_faces_list.append(face_verts)
    
    # Generate OBB for wall faces (Layer 2)
    if len(wall_faces) > 0:
        layer_faces = wall_faces
        layer_normals = wall_normals
        all_verts = np.vstack(layer_faces)
        avg_normal = np.mean(layer_normals, axis=0)
        avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-10)
        
        obb_params = create_obb(all_verts, avg_normal)
        obb_faces = obb_to_face_vertices(obb_params)
        
        for face_idx, face_verts in enumerate(obb_faces):
            simplified_cat.append(0)  # Face type is 0
            simplified_idd.append(f'layer2_obb_face_{face_idx}')
            
            v1 = face_verts[1] - face_verts[0]
            v2 = face_verts[3] - face_verts[0]
            face_normal = np.cross(v1, v2)
            face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-10)
            simplified_normal.append(face_normal)
            simplified_faces_list.append(face_verts)
    
    # If no faces were categorized, return single OBB
    if len(simplified_faces_list) == 0:
        return _simplify_to_single_obb(cat, idd, normal, faces)
    
    return (
        np.array(simplified_cat),
        simplified_idd,
        np.array(simplified_normal),
        simplified_faces_list
    )