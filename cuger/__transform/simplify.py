"""
    This module provides functions to simplify a geo file to different levels of detail. The main function is `simplify_faces`, which takes a batch of faces and produces a simplified version based on the specified level of detail (LOD). The LOD can be set to "precise", "medium", or "low", which determines the degree of simplification applied to the geometry.
    
    - precise: output original geometry without simplification
    - medium: output multi-layer OBB extraction based on embedding layers
    - low: output the entire OBB as a single simplified geometry
"""

import numpy as np
from .geometry import create_obb, obb_to_face_vertices, calculate_wwr, GeometryBasic


def _polygon_area_3d(vertices):
    if vertices is None or len(vertices) < 3:
        return 0.0

    verts = np.asarray(vertices, dtype=float)
    p0 = verts[0]
    area = 0.0
    for i in range(1, len(verts) - 1):
        e1 = verts[i] - p0
        e2 = verts[i + 1] - p0
        area += 0.5 * np.linalg.norm(np.cross(e1, e2))
    return float(area)


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
        (simplified_cat, simplified_idd, simplified_normal, simplified_faces, simplified_holes)
    """
    
    if lod == "precise":
        # Return original geometry without simplification
        return cat, idd, normal, faces, holes
    
    # Calculate WWR from original geometry for low and medium LOD
    wwr = calculate_wwr(cat, faces, normal)
    
    if lod == "low":
        # Return entire OBB as single simplified geometry
        return _simplify_to_single_obb(cat, idd, normal, faces, wwr)
    
    elif lod == "medium":
        # Return multi-layer OBB extraction
        return _simplify_to_multi_layer_obb(cat, idd, normal, faces, wwr)
    
    else:
        # Default if unknown LOD
        return cat, idd, normal, faces, holes


def _create_window_on_wall(face_verts, face_normal, wwr, margin_ratio=0.1):
    """
    Create a window opening on a wall face based on WWR.
    
    Parameters
    ----------
    face_verts : np.ndarray
        Vertices of the wall face (4 vertices).
    face_normal : np.ndarray
        Normal vector of the wall face.
    wwr : float
        Window-to-wall ratio (0.0 - 1.0).
    margin_ratio : float
        Margin ratio from edges (default 0.1 = 10% margin).
    
    Returns
    -------
    tuple
        (hole_verts, window_verts) - hole vertices and window face vertices
    """
    if wwr <= 0 or abs(face_normal[2]) > 0.7:
        # No window for non-walls or zero WWR
        return None, None
    
    # Calculate face center and local coordinate system
    center = np.mean(face_verts, axis=0)
    
    # Create local coordinate system
    v1 = face_verts[1] - face_verts[0]
    v2 = face_verts[3] - face_verts[0]
    
    # Normalize vectors
    u = v1 / (np.linalg.norm(v1) + 1e-10)
    v = v2 / (np.linalg.norm(v2) + 1e-10)
    
    # Get wall dimensions
    width = np.linalg.norm(v1)
    height = np.linalg.norm(v2)
    
    # Calculate window dimensions based on WWR
    # Window area = WWR * Wall area
    # Assuming rectangular window, keep aspect ratio similar to wall
    window_scale = np.sqrt(wwr)
    window_width = width * window_scale * (1 - 2 * margin_ratio)
    window_height = height * window_scale * (1 - 2 * margin_ratio)
    
    # Create window vertices centered on wall
    half_w = window_width / 2
    half_h = window_height / 2
    
    window_verts = np.array([
        center - half_w * u - half_h * v,
        center + half_w * u - half_h * v,
        center + half_w * u + half_h * v,
        center - half_w * u + half_h * v,
    ])
    
    # Return holes as dict format: {hole_id: vertices}
    hole_dict = {0: window_verts}
    
    return hole_dict, window_verts


def _simplify_to_single_obb(cat, idd, normal, faces, wwr):
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
    wwr : float
        Window-to-wall ratio.
    
    Returns
    -------
    tuple
        (simplified_cat, simplified_idd, simplified_normal, simplified_faces, simplified_holes)
    """
    # Combine all vertices into single point cloud
    all_verts = np.vstack(faces)
    
    # Calculate average normal (or use z-axis if all normals are similar)
    normal = (0,0,1)

    # Create OBB for all points
    obb_params = create_obb(all_verts, normal)

    obb_faces = obb_to_face_vertices(obb_params)
    
    # Generate category, ID, normal, holes, and window faces for OBB faces
    simplified_cat = []
    simplified_idd = []
    simplified_normal = []
    simplified_faces_list = []
    simplified_holes = []
    
    for i, face_verts in enumerate(obb_faces):
        # Calculate normal from vertices
        v1 = face_verts[1] - face_verts[0]
        v2 = face_verts[3] - face_verts[0]
        face_normal = np.cross(v1, v2)
        face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-10)
        
        # Create window opening on wall faces
        hole, window_face = _create_window_on_wall(face_verts, face_normal, wwr)
        
        # Add wall face with hole
        simplified_cat.append(0)  # Wall category = 0
        simplified_idd.append(f'obb_face_{i}')
        simplified_normal.append(face_normal)
        simplified_faces_list.append(face_verts)
        simplified_holes.append(hole)
        
        # Add window face if created
        if window_face is not None:
            simplified_cat.append(1)  # Window category = 1
            simplified_idd.append(f'obb_window_{i}')
            simplified_normal.append(face_normal)
            simplified_faces_list.append(window_face)
            simplified_holes.append(None)
    
    return (
        np.array(simplified_cat),
        simplified_idd,
        np.array(simplified_normal),
        simplified_faces_list,
        simplified_holes
    )


def _simplify_to_multi_layer_obb(cat, idd, normal, faces, wwr):
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
    wwr : float
        Window-to-wall ratio.
    
    Returns
    -------
    tuple
        (simplified_cat, simplified_idd, simplified_normal, simplified_faces, simplified_holes)
    """
    simplified_cat = []
    simplified_idd = []
    simplified_normal = []
    simplified_faces_list = []
    simplified_holes = []
    
    if len(faces) == 0:
        return (np.array([]), [], np.array([]), [], [])

    min_face_area = 1e-5
    min_span = 1e-4
    min_layer_height = 1e-3
    unique_round_decimals = 6
    
    # Collect floor/roof-like faces first (normal mostly along Z axis)
    floor_faces = []
    for i, face_normal in enumerate(normal):
        if abs(face_normal[2]) > 0.7:
            candidate_face = np.asarray(faces[i], dtype=float)
            if len(candidate_face) < 3:
                continue
            if _polygon_area_3d(candidate_face) <= min_face_area:
                continue
            floor_faces.append(candidate_face)
    
    if len(floor_faces) == 0:
        return _simplify_to_single_obb(cat, idd, normal, faces, wwr)

    # Group floor faces by their Z level (centroid Z), then build OBB per layer.
    # A layer is defined by two adjacent Z levels.
    z_centers = [float(np.mean(np.asarray(face)[:, 2])) for face in floor_faces]
    z_range = max(z_centers) - min(z_centers)
    z_tol = max(0.05, z_range * 0.01)

    sorted_pairs = sorted(zip(z_centers, floor_faces), key=lambda item: item[0])
    z_groups = []
    for zc, face in sorted_pairs:
        if not z_groups:
            z_groups.append({'z_values': [zc], 'faces': [face]})
            continue
        current_mean_z = float(np.mean(z_groups[-1]['z_values']))
        if abs(zc - current_mean_z) <= z_tol:
            z_groups[-1]['z_values'].append(zc)
            z_groups[-1]['faces'].append(face)
        else:
            z_groups.append({'z_values': [zc], 'faces': [face]})

    level_z = [float(np.mean(group['z_values'])) for group in z_groups]

    # Need at least two levels to infer story height from floor faces.
    if len(level_z) < 2:
        return _simplify_to_single_obb(cat, idd, normal, faces, wwr)

    for layer_idx in range(len(level_z) - 1):
        bottom_z = level_z[layer_idx]
        top_z = level_z[layer_idx + 1]
        layer_height = top_z - bottom_z

        if layer_height <= min_layer_height:
            continue

        layer_faces = z_groups[layer_idx]['faces']
        all_verts = np.vstack(layer_faces)

        span_x = float(np.ptp(all_verts[:, 0]))
        span_y = float(np.ptp(all_verts[:, 1]))
        if span_x <= min_span or span_y <= min_span:
            continue

        unique_xy = np.unique(np.round(all_verts[:, :2], unique_round_decimals), axis=0)
        if len(unique_xy) < 3:
            continue

        # Build footprint OBB from current level floor faces, then assign layer height.
        obb_params = create_obb(all_verts, (0, 0, 1))
        obb_params['scale'][2] = layer_height
        obb_params['center'][2] = bottom_z + layer_height / 2.0

        obb_faces = obb_to_face_vertices(obb_params)

        for face_idx, face_verts in enumerate(obb_faces):
            if _polygon_area_3d(face_verts) <= min_face_area:
                continue

            v1 = face_verts[1] - face_verts[0]
            v2 = face_verts[3] - face_verts[0]
            current_face_normal = np.cross(v1, v2)
            current_face_normal = current_face_normal / (np.linalg.norm(current_face_normal) + 1e-10)
            
            # Create window opening on wall faces
            hole, window_face = _create_window_on_wall(face_verts, current_face_normal, wwr)
            
            # Add wall face with hole
            simplified_cat.append(0)  # Wall category = 0
            simplified_idd.append(f'layer{layer_idx + 1}_obb_face_{face_idx}')
            simplified_normal.append(current_face_normal)
            simplified_faces_list.append(face_verts)
            simplified_holes.append(hole)
            
            # Add window face if created
            if window_face is not None:
                simplified_cat.append(1)  # Window category = 1
                simplified_idd.append(f'layer{layer_idx + 1}_obb_window_{face_idx}')
                simplified_normal.append(current_face_normal)
                simplified_faces_list.append(window_face)
                simplified_holes.append(None)
    
    # Fallback if no valid layer OBB could be created.
    if len(simplified_faces_list) == 0:
        return _simplify_to_single_obb(cat, idd, normal, faces, wwr)
    
    return (
        np.array(simplified_cat),
        simplified_idd,
        np.array(simplified_normal),
        simplified_faces_list,
        simplified_holes
    )