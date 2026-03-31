"""
    This module provides functions to simplify a geo file to different levels of detail. The main function is `simplify_faces`, which takes a batch of faces and produces a simplified version based on the specified level of detail (LOD). The LOD can be set to "precise", "medium", or "low", which determines the degree of simplification applied to the geometry.
    
    - precise: output original geometry without simplification
    - medium: output multi-layer OBB extraction based on embedding layers
    - low: output the entire OBB as a single simplified geometry
"""

import math

import numpy as np
import pygeos
from .geometry import create_obb, obb_to_face_vertices, calculate_wwr, GeometryBasic, GeometryOperator


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


def _group_horizontal_face_levels(faces, normal, min_face_area=1e-5):
    horizontal = []
    for idx, face_normal in enumerate(normal):
        if np.abs(face_normal[2]) <= 0.7:
            continue

        face = np.asarray(faces[idx], dtype=float)
        if len(face) < 3:
            continue
        if GeometryBasic.polygon_area_3d(face) <= min_face_area:
            continue

        horizontal.append((float(np.mean(face[:, 2])), idx))

    if not horizontal:
        return []

    z_values = [item[0] for item in horizontal]
    z_range = max(z_values) - min(z_values)
    z_tol = max(0.05, z_range * 0.01)

    groups = []
    for z_center, idx in sorted(horizontal, key=lambda item: item[0]):
        if not groups:
            groups.append({"z_values": [z_center], "indices": [idx]})
            continue

        current_mean = float(np.mean(groups[-1]["z_values"]))
        if abs(z_center - current_mean) <= z_tol:
            groups[-1]["z_values"].append(z_center)
            groups[-1]["indices"].append(idx)
        else:
            groups.append({"z_values": [z_center], "indices": [idx]})

    return groups


def _compute_dominant_xy_axes(faces, edge_tol=1e-6):
    orient_x = 0.0
    orient_y = 0.0

    for face in faces:
        verts = np.asarray(face, dtype=float)
        if len(verts) < 2:
            continue

        for idx in range(len(verts)):
            edge = verts[(idx + 1) % len(verts), :2] - verts[idx, :2]
            length = float(np.linalg.norm(edge))
            if length <= edge_tol:
                continue

            theta = math.atan2(edge[1], edge[0])
            orient_x += length * math.cos(4.0 * theta)
            orient_y += length * math.sin(4.0 * theta)

    if abs(orient_x) <= edge_tol and abs(orient_y) <= edge_tol:
        angle = 0.0
    else:
        angle = 0.25 * math.atan2(orient_y, orient_x)

    x_axis = np.array([math.cos(angle), math.sin(angle)], dtype=float)
    y_axis = np.array([-math.sin(angle), math.cos(angle)], dtype=float)
    return x_axis, y_axis


def _build_rect_face(center_xy, x_axis, y_axis, span_x, span_y, z_value):
    half_x = 0.5 * span_x
    half_y = 0.5 * span_y
    xy = np.array(
        [
            center_xy - half_x * x_axis - half_y * y_axis,
            center_xy + half_x * x_axis - half_y * y_axis,
            center_xy + half_x * x_axis + half_y * y_axis,
            center_xy - half_x * x_axis + half_y * y_axis,
        ],
        dtype=float,
    )

    z_column = np.full((4, 1), float(z_value), dtype=float)
    return np.hstack((xy, z_column))


def _face_to_xy_polygon(face):
    verts = np.asarray(face, dtype=float)
    if len(verts) < 3:
        return None

    xy = verts[:, :2]
    if np.linalg.norm(xy[0] - xy[-1]) > 1e-8:
        xy = np.vstack((xy, xy[0]))

    polygon = pygeos.polygons(xy)
    if pygeos.is_empty(polygon) or pygeos.get_dimensions(polygon) != 2:
        return None
    return polygon


def _largest_polygon_part(geometry):
    if geometry is None or pygeos.is_empty(geometry):
        return None

    if pygeos.get_type_id(geometry) == 3:
        return geometry

    parts = [
        part
        for part in pygeos.get_parts(geometry)
        if pygeos.get_dimensions(part) == 2 and not pygeos.is_empty(part)
    ]
    if not parts:
        return None

    return max(parts, key=lambda part: float(pygeos.area(part)))


def _intersect_xy_polygons(polygons):
    if not polygons:
        return None

    intersection = polygons[0]
    for polygon in polygons[1:]:
        intersection = pygeos.intersection(intersection, polygon)
        if pygeos.is_empty(intersection) or pygeos.get_dimensions(intersection) != 2:
            return None

    polygon = _largest_polygon_part(intersection)
    if polygon is None or float(pygeos.area(polygon)) <= 1e-6:
        return None
    return polygon


def _project_polygon_bounds(polygon, x_axis, y_axis):
    coords = np.asarray(pygeos.get_coordinates(pygeos.get_exterior_ring(polygon)), dtype=float)[:-1, :2]
    proj_x = coords @ x_axis
    proj_y = coords @ y_axis
    return (
        float(np.min(proj_x)),
        float(np.max(proj_x)),
        float(np.min(proj_y)),
        float(np.max(proj_y)),
    )


def _build_rect_polygon(center_xy, x_axis, y_axis, span_x, span_y):
    half_x = 0.5 * span_x
    half_y = 0.5 * span_y
    rect_xy = np.array(
        [
            center_xy - half_x * x_axis - half_y * y_axis,
            center_xy + half_x * x_axis - half_y * y_axis,
            center_xy + half_x * x_axis + half_y * y_axis,
            center_xy - half_x * x_axis + half_y * y_axis,
        ],
        dtype=float,
    )
    return pygeos.polygons(np.vstack((rect_xy, rect_xy[0])))


def _projected_center_to_world(min_x, max_x, min_y, max_y, x_axis, y_axis):
    center_local = np.array(
        [
            0.5 * (min_x + max_x),
            0.5 * (min_y + max_y),
        ],
        dtype=float,
    )
    return center_local[0] * x_axis + center_local[1] * y_axis


def _fit_rect_in_polygon(polygon, x_axis, y_axis, target_area, preferred_center_xy=None, min_scale=0.92):
    if polygon is None or target_area <= 1e-6:
        return None

    min_x, max_x, min_y, max_y = _project_polygon_bounds(polygon, x_axis, y_axis)
    span_x = max_x - min_x
    span_y = max_y - min_y
    if span_x <= 1e-6 or span_y <= 1e-6:
        return None

    center_xy = None
    if preferred_center_xy is not None:
        preferred_center_xy = np.asarray(preferred_center_xy, dtype=float)
        if preferred_center_xy.shape == (2,):
            if bool(pygeos.contains(polygon, pygeos.points(preferred_center_xy))):
                center_xy = preferred_center_xy

    if center_xy is None:
        center_xy = _projected_center_to_world(min_x, max_x, min_y, max_y, x_axis, y_axis)
    center_point = pygeos.points(center_xy)
    if not bool(pygeos.contains(polygon, center_point)):
        center_geom = pygeos.centroid(polygon)
        if not bool(pygeos.contains(polygon, center_geom)):
            center_geom = pygeos.point_on_surface(polygon)
        center_xy = np.asarray(pygeos.get_coordinates(center_geom), dtype=float)[0, :2]

    aspect_ratio = max(span_x / max(span_y, 1e-8), 1e-8)
    rect_span_x = math.sqrt(target_area * aspect_ratio)
    rect_span_y = target_area / max(rect_span_x, 1e-8)

    bbox_scale = min(
        1.0,
        span_x / max(rect_span_x, 1e-8),
        span_y / max(rect_span_y, 1e-8),
    )
    rect_span_x *= bbox_scale
    rect_span_y *= bbox_scale

    def _contains(scale):
        if scale <= 1e-8:
            return False
        rect_polygon = _build_rect_polygon(
            center_xy,
            x_axis,
            y_axis,
            rect_span_x * scale,
            rect_span_y * scale,
        )
        return bool(pygeos.contains(polygon, rect_polygon))

    low = 0.0
    high = 1.0
    if _contains(high):
        low = high
    else:
        for _ in range(40):
            mid = 0.5 * (low + high)
            if _contains(mid):
                low = mid
            else:
                high = mid

    final_scale = low * min_scale
    if final_scale <= 1e-6:
        return None

    return center_xy, rect_span_x * final_scale, rect_span_y * final_scale


def _append_face_hole(face_holes, hole_face):
    hole_face = np.asarray(hole_face, dtype=float)
    if face_holes is None:
        return {0: hole_face}

    ordered_holes = [np.asarray(face_holes[key], dtype=float) for key in sorted(face_holes)]
    ordered_holes.append(hole_face)
    return {idx: hole for idx, hole in enumerate(ordered_holes)}


def _build_core_wall_faces(bottom_face, top_face, core_center_xy):
    wall_faces = []
    wall_normals = []

    for idx in range(len(bottom_face)):
        face = np.array(
            [
                bottom_face[idx],
                top_face[idx],
                top_face[(idx + 1) % len(top_face)],
                bottom_face[(idx + 1) % len(bottom_face)],
            ],
            dtype=float,
        )

        v1 = face[1] - face[0]
        v2 = face[3] - face[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm <= 1e-8:
            continue
        normal = normal / norm

        face_center_xy = np.mean(face[:, :2], axis=0)
        outward_xy = face_center_xy - core_center_xy
        if np.dot(normal[:2], outward_xy) < 0:
            face = face[::-1]
            v1 = face[1] - face[0]
            v2 = face[3] - face[0]
            normal = np.cross(v1, v2)
            normal = normal / (np.linalg.norm(normal) + 1e-12)

        wall_faces.append(face)
        wall_normals.append(normal)

    return wall_faces, wall_normals


def inject_minimal_core(cat, idd, normal, faces, holes, core_area_ratio=0.2, lod=None):
    """
    Inject a simple rectangular core shaft into low/medium simplified geometry.
    """
    level_groups = _group_horizontal_face_levels(faces, normal)
    if len(level_groups) < 2:
        print("--Minimal core skipped: insufficient horizontal levels--")
        return cat, idd, normal, faces, holes

    x_axis, y_axis = _compute_dominant_xy_axes(faces)

    level_info = []
    for group in level_groups:
        group_polygons = []
        for idx in group["indices"]:
            polygon = _face_to_xy_polygon(faces[idx])
            if polygon is not None:
                group_polygons.append(polygon)

        common_polygon = _intersect_xy_polygons(group_polygons)
        if common_polygon is None:
            continue

        level_area = float(pygeos.area(common_polygon))
        min_x, max_x, min_y, max_y = _project_polygon_bounds(common_polygon, x_axis, y_axis)
        level_center = _projected_center_to_world(min_x, max_x, min_y, max_y, x_axis, y_axis)
        level_info.append(
            {
                "z": float(np.mean(group["z_values"])),
                "indices": list(group["indices"]),
                "polygon": common_polygon,
                "center": level_center,
                "area": float(level_area),
            }
        )

    if not level_info:
        print("--Minimal core skipped: invalid horizontal reference area--")
        return cat, idd, normal, faces, holes

    if lod == "medium":
        centered_candidates = []
        fallback_candidates = []

        for start in range(len(level_info) - 1):
            for end in range(start + 1, len(level_info)):
                selected_candidate_levels = level_info[start : end + 1]
                candidate_polygon = _intersect_xy_polygons(
                    [level["polygon"] for level in selected_candidate_levels]
                )
                if candidate_polygon is None:
                    continue

                available_area = float(pygeos.area(candidate_polygon))
                story_count = end - start
                average_candidate_area = float(
                    np.mean([level["area"] for level in selected_candidate_levels])
                )
                target_candidate_area = average_candidate_area * float(core_area_ratio)
                rect_candidate_area = min(target_candidate_area, available_area)
                preferred_candidate_center = np.mean(
                    [level["center"] for level in selected_candidate_levels],
                    axis=0,
                )

                candidate = (
                    story_count,
                    rect_candidate_area,
                    available_area,
                    (start, end),
                    candidate_polygon,
                    selected_candidate_levels,
                    preferred_candidate_center,
                    target_candidate_area,
                )
                fallback_candidates.append(candidate)

                if bool(pygeos.contains(candidate_polygon, pygeos.points(preferred_candidate_center))):
                    centered_candidates.append(candidate)

        candidate_pool = centered_candidates or fallback_candidates
        if not candidate_pool:
            print("--Minimal core skipped: no shared footprint for medium levels--")
            return cat, idd, normal, faces, holes

        (
            best_story_count,
            rect_area,
            best_available_area,
            best_range,
            best_polygon,
            selected_levels,
            preferred_center_xy,
            target_area,
        ) = max(candidate_pool, key=lambda item: (item[0], item[1], item[2]))
        range_label = f"levels {best_range[0] + 1}-{best_range[1] + 1}"
    else:
        reference_area = max(level["area"] for level in level_info)
        target_area = reference_area * float(core_area_ratio)
        centered_candidates = []
        fallback_candidates = []

        for start in range(len(level_info) - 1):
            for end in range(start + 1, len(level_info)):
                selected_candidate_levels = level_info[start : end + 1]
                candidate_polygon = _intersect_xy_polygons(
                    [level["polygon"] for level in selected_candidate_levels]
                )
                if candidate_polygon is None:
                    continue

                available_area = float(pygeos.area(candidate_polygon))
                story_count = end - start
                rect_candidate_area = min(target_area, available_area)
                preferred_candidate_center = np.mean(
                    [level["center"] for level in selected_candidate_levels],
                    axis=0,
                )

                candidate = (
                    story_count,
                    rect_candidate_area,
                    available_area,
                    (start, end),
                    candidate_polygon,
                    selected_candidate_levels,
                    preferred_candidate_center,
                )
                fallback_candidates.append(candidate)

                if bool(pygeos.contains(candidate_polygon, pygeos.points(preferred_candidate_center))):
                    centered_candidates.append(candidate)

        candidate_pool = centered_candidates or fallback_candidates
        if not candidate_pool:
            print("--Minimal core skipped: no shared footprint for any story stack--")
            return cat, idd, normal, faces, holes

        (
            best_story_count,
            rect_area,
            best_available_area,
            best_range,
            best_polygon,
            selected_levels,
            preferred_center_xy,
        ) = max(candidate_pool, key=lambda item: (item[0], item[1], item[2]))
        range_label = f"levels {best_range[0] + 1}-{best_range[1] + 1}"

    if rect_area <= 1e-6:
        print("--Minimal core skipped: target core area too small--")
        return cat, idd, normal, faces, holes

    fitted_rect = _fit_rect_in_polygon(
        best_polygon,
        x_axis,
        y_axis,
        rect_area,
        preferred_center_xy=preferred_center_xy,
    )
    if fitted_rect is None:
        print("--Minimal core skipped: unable to place core inside shared footprint--")
        return cat, idd, normal, faces, holes

    center_xy, rect_span_x, rect_span_y = fitted_rect
    actual_rect_area = rect_span_x * rect_span_y
    if actual_rect_area <= 1e-6:
        print("--Minimal core skipped: fitted core area too small--")
        return cat, idd, normal, faces, holes

    if actual_rect_area < rect_area:
        rect_area = actual_rect_area

    core_cat = list(cat)
    core_idd = list(idd)
    core_normal = [np.asarray(n, dtype=float) for n in normal]
    core_faces = [np.asarray(face, dtype=float) for face in faces]
    core_holes = list(holes)

    selected_indices = [idx for level in selected_levels for idx in level["indices"]]

    for idx in selected_indices:
        z_value = float(np.mean(core_faces[idx][:, 2]))
        hole_face = _build_rect_face(center_xy, x_axis, y_axis, rect_span_x, rect_span_y, z_value)
        hole_face = GeometryOperator.reorder_vertices(hole_face, is_upward=False)
        core_holes[idx] = _append_face_hole(core_holes[idx], hole_face)

    level_z = [level["z"] for level in selected_levels]
    for story_idx in range(len(level_z) - 1):
        bottom_z = level_z[story_idx]
        top_z = level_z[story_idx + 1]
        if top_z - bottom_z <= 1e-6:
            continue

        bottom_face = _build_rect_face(center_xy, x_axis, y_axis, rect_span_x, rect_span_y, bottom_z)
        top_face = _build_rect_face(center_xy, x_axis, y_axis, rect_span_x, rect_span_y, top_z)
        wall_faces, wall_normals = _build_core_wall_faces(bottom_face, top_face, center_xy)
        story_label = best_range[0] + story_idx + 1

        for wall_idx, wall_face in enumerate(wall_faces):
            core_cat.append("2")
            core_idd.append(f"core_wall_{story_label}_{wall_idx}")
            core_normal.append(np.asarray(wall_normals[wall_idx], dtype=float))
            core_faces.append(wall_face)
            core_holes.append(None)

    print(
        f"--Minimal core inserted across {max(len(level_z) - 1, 0)} layers "
        f"({range_label})--"
    )
    return (
        np.array(core_cat),
        core_idd,
        np.array(core_normal),
        core_faces,
        core_holes,
    )


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

    obb_faces, obb_normals = obb_to_face_vertices(obb_params)
    
    # Generate category, ID, normal, holes, and window faces for OBB faces
    simplified_cat = []
    simplified_idd = []
    simplified_normal = []
    simplified_faces_list = []
    simplified_holes = []
    
    for i, face_verts in enumerate(obb_faces):
        face_normal = obb_normals[i]
        
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

    # Collect wall-like faces once (normal mostly horizontal) for per-layer OBB fitting.
    wall_faces = []
    for i, face_normal in enumerate(normal):
        if abs(face_normal[2]) <= 0.7:
            candidate_face = np.asarray(faces[i], dtype=float)
            if len(candidate_face) < 3:
                continue
            if _polygon_area_3d(candidate_face) <= min_face_area:
                continue
            wall_faces.append(candidate_face)

    for layer_idx in range(len(level_z) - 1):
        bottom_z = level_z[layer_idx]
        top_z = level_z[layer_idx + 1]
        layer_height = top_z - bottom_z

        if layer_height <= min_layer_height:
            continue

        z_band_tol = max(1e-4, layer_height * 0.02)

        # Prefer wall vertices whose face centroid z falls in current story band.
        layer_wall_verts = []
        for wall_face in wall_faces:
            face_z_center = float(np.mean(wall_face[:, 2]))

            # Keep wall faces by centroid-z membership in the current layer.
            if face_z_center < bottom_z - z_band_tol or face_z_center > top_z + z_band_tol:
                continue

            layer_wall_verts.append(wall_face)

        if len(layer_wall_verts) > 0:
            all_verts = np.vstack(layer_wall_verts)
        else:
            # Fallback to floor vertices when wall faces are unavailable.
            layer_faces = z_groups[layer_idx]['faces']
            all_verts = np.vstack(layer_faces)

        span_x = float(np.ptp(all_verts[:, 0]))
        span_y = float(np.ptp(all_verts[:, 1]))
        if span_x <= min_span or span_y <= min_span:
            continue

        unique_xy = np.unique(np.round(all_verts[:, :2], unique_round_decimals), axis=0)
        if len(unique_xy) < 3:
            continue

        # Fit OBB from layer vertices (wall-priority), then assign story height.
        obb_params = create_obb(all_verts, (0, 0, 1))
        obb_params['scale'][2] = layer_height
        obb_params['center'][2] = bottom_z + layer_height / 2.0

        obb_faces, obb_normals = obb_to_face_vertices(obb_params)

        for face_idx, face_verts in enumerate(obb_faces):
            if _polygon_area_3d(face_verts) <= min_face_area:
                continue
            current_face_normal = obb_normals[face_idx]
            
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
