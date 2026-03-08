"""
Geometry utilities module for convexification and simplification operations.
Contains basic geometry classes, validators, and operators.
"""

import numpy as np
import pygeos
from typing import Union, List, Tuple, Optional
from collections import defaultdict
from scipy.spatial.transform import Rotation as R


# ============================================================================
# GEOMETRY BASIC CLASS
# ============================================================================

class GeometryBasic:
    """
    Basic geometry utility methods.
    All methods in this class are static and provide fundamental geometric calculations.
    """
    @staticmethod
    def angle(p1, p2, p3):
        """Calculate signed angle at p2 formed by p1-p2-p3."""
        v1 = p2 - p1
        v2 = p3 - p2
        cross = np.cross(v1, v2)
        dot = np.dot(v1, v2)
        
        if np.linalg.norm(cross) < 1e-3 * np.linalg.norm(v1) * np.linalg.norm(v2):
            return 0  
        
        angle_rad = np.arccos(np.clip(dot/(np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg if cross[2] > 0 else -angle_deg

    @staticmethod
    def get_angle_tan(p1, p2, verts_all):
        vec = verts_all[p2] - verts_all[p1]
        return np.arctan2(vec[1], vec[0])

    @staticmethod
    def polygon_area_2d(vertices):
        """
        Calculate the area of a 2D polygon using the shoelace formula.
        
        Parameters
        ----------
        vertices : list or numpy.ndarray
            List of 2D vertices (x, y) forming a polygon.
        
        Returns
        -------
        float
            The area of the polygon.
        """
        if len(vertices) < 3:
            return 0.0
        
        area = 0.0
        n = len(vertices)
        for i in range(n):
            x1, y1 = vertices[i][:2]
            x2, y2 = vertices[(i + 1) % n][:2]
            area += x1 * y2 - x2 * y1
        
        return abs(area) / 2.0
    
    @staticmethod
    def polygon_area_3d(vertices):
        """
        Calculate the area of a 3D polygon by projecting it onto the best-fit plane.
        
        Parameters
        ----------
        vertices : list or numpy.ndarray
            List of 3D vertices (x, y, z) forming a polygon.
        
        Returns
        -------
        float
            The area of the polygon.
        """
        if len(vertices) < 3:
            return 0.0
        
        v = np.asarray(vertices, dtype=float)
        
        # Compute normal vector using Newell's method
        normal = np.zeros(3)
        for i in range(len(v)):
            current = v[i]
            next_vertex = v[(i + 1) % len(v)]
            normal[0] += (current[1] - next_vertex[1]) * (current[2] + next_vertex[2])
            normal[1] += (current[2] - next_vertex[2]) * (current[0] + next_vertex[0])
            normal[2] += (current[0] - next_vertex[0]) * (current[1] + next_vertex[1])
        
        area = np.linalg.norm(normal) / 2.0
        return area

# ============================================================================
# GEOMETRY VALIDATOR CLASS
# ============================================================================

class GeometryValidator:
    """
    Geometry validation and checking methods.
    All methods in this class return boolean values or perform geometric tests.
    """
    
    @staticmethod
    def _is_left_on(p1, p2, p3):
        """Check if p3 is on the left side of line p1->p2 in 2D."""
        p1_2d = p1[:2]
        p2_2d = p2[:2]
        p3_2d = p3[:2]
        cross = np.cross(p2_2d - p1_2d, p3_2d - p2_2d)
        if cross > 0 and np.abs(cross) < 1e-6 * np.linalg.norm(p2_2d - p1_2d) * np.linalg.norm(p3_2d - p2_2d):
            return False
        return cross > 0
    
    @staticmethod
    def _is_collinear(p1, p2, p3):
        """Check if three points are collinear."""
        area = np.cross(p2 - p1, p3 - p2)
        dist = area / (np.dot(p1, p2 - p1) + 1e-6)
        return np.abs(dist) < 1e-3

    @staticmethod
    def _is_between(p1, p2, p3):
        """Check if p3 is between p1 and p2."""
        p1_2d = p1[:2]
        p2_2d = p2[:2]
        p3_2d = p3[:2]
        if p1_2d[0] != p2_2d[0]:
            return (p1_2d[0] <= p3_2d[0] <= p2_2d[0]) or (p1_2d[0] >= p3_2d[0] >= p2_2d[0])
        else:
            return (p1_2d[1] <= p3_2d[1] <= p2_2d[1]) or (p1_2d[1] >= p3_2d[1] >= p2_2d[1])

    @staticmethod
    def _is_intersect(a, b, c, d):
        """Check if line segment ab intersects with line segment cd."""
        a_2d = a[:2]
        b_2d = b[:2]
        c_2d = c[:2]
        d_2d = d[:2]
        if GeometryValidator._is_collinear(a_2d, b_2d, c_2d):
            return GeometryValidator._is_between(a_2d, b_2d, c_2d)
        if GeometryValidator._is_collinear(a_2d, b_2d, d_2d):
            return GeometryValidator._is_between(a_2d, b_2d, d_2d)
        if GeometryValidator._is_collinear(c_2d, d_2d, a_2d):
            return GeometryValidator._is_between(c_2d, d_2d, a_2d)
        if GeometryValidator._is_collinear(c_2d, d_2d, b_2d):
            return GeometryValidator._is_between(c_2d, d_2d, b_2d)
        cd_cross = np.logical_xor(GeometryValidator._is_left_on(a_2d, b_2d, c_2d), GeometryValidator._is_left_on(a_2d, b_2d, d_2d))
        ab_cross = np.logical_xor(GeometryValidator._is_left_on(c_2d, d_2d, a_2d), GeometryValidator._is_left_on(c_2d, d_2d, b_2d))
        return ab_cross and cd_cross
    
    @staticmethod
    def _is_obtuse(v1, v2, v3):
        """Check if angle at v2 is obtuse."""
        return GeometryBasic.angle(v1, v2, v3) > 90
    
    @staticmethod
    def _is_valid_face(vertices, area_eps=1e-8):
        """
        Check whether a 3D polygon face is geometrically valid.

        Parameters
        ----------
        vertices : list or np.ndarray, shape (N, 3)
            Vertices of the polygon.
        area_eps : float
            Area threshold below which the face is considered degenerate.

        Returns
        -------
        bool
            True if face is valid, False otherwise.
        """
        # 1. Vertex count
        if vertices is None or len(vertices) < 3:
            return False

        v = np.asarray(vertices, dtype=float)

        # 2. Finite check
        if not np.isfinite(v).all():
            return False

        # 3. Area check (fan triangulation)
        p0 = v[0]
        area = 0.0

        for i in range(1, len(v) - 1):
            e1 = v[i]     - p0
            e2 = v[i + 1] - p0
            area += 0.5 * np.linalg.norm(np.cross(e1, e2))

        if area <= area_eps:
            return False

        return True
    
    @staticmethod
    def _is_same_polygon(polygon1, polygon2, projection=False):
        """
        Determine whether two polygons are identical.

        Args:
            polygon1: numpy array of shape (n, 2) or (n, 3)
            polygon2: numpy array with the same shape as polygon1
            projection: bool, if True, comparison is performed only on XY projection

        Returns:
            bool: True if the two polygons are identical, False otherwise
        """
        if polygon1.shape != polygon2.shape:
            return False

        if projection:
            if polygon1.shape[1] < 2 or polygon2.shape[1] < 2:
                return False
            poly1 = polygon1[:, :2]
            poly2 = polygon2[:, :2]
        else:
            poly1 = polygon1
            poly2 = polygon2

        # Direct match
        if np.array_equal(poly1, poly2):
            return True

        # First point same, others reversed
        if np.array_equal(poly1[0], poly2[0]) and np.array_equal(poly1[1:], poly2[1:][::-1]):
            return True

        return False
    
    @staticmethod
    def _is_diagonal(verts: np.ndarray, indices: np.ndarray, ia: int, ib: int) -> bool:
        """Check if diagonal between two vertices is valid."""
        def in_cone(verts: np.ndarray, indices: np.ndarray, ia: int, ib: int) -> bool:
            n = len(indices)
            ia_prev = ia - 1 if ia - 1 >= 0 else n - 1
            ia_next = ia + 1 if ia + 1 < n else 0

            ia, ib = indices[ia], indices[ib]
            ia_prev, ia_next = indices[ia_prev], indices[ia_next]

            # Convex
            if GeometryValidator._is_left_on(verts[ia_prev], verts[ia], verts[ia_next]):
                return GeometryValidator._is_left_on(verts[ia], verts[ib], verts[ia_prev]) and \
                    GeometryValidator._is_left_on(verts[ib], verts[ia], verts[ia_next])
            # Concave
            return not (GeometryValidator._is_left_on(verts[ia], verts[ib], verts[ia_next]) and \
                        GeometryValidator._is_left_on(verts[ib], verts[ia], verts[ia_prev]))

        def diagonalie(verts: np.ndarray, indices: np.ndarray, ia: int, ib: int) -> bool:
            n = len(indices)
            for now_i in range(n):
                if indices[now_i] == indices[ia] or indices[now_i] == indices[ib]:
                    continue
                next_i = (now_i + 1) % n
                if indices[next_i] == indices[ia] or indices[next_i] == indices[ib]:
                    continue

                if GeometryValidator._is_intersect(
                        verts[indices[ia]], verts[indices[ib]],
                        verts[indices[now_i]], verts[indices[next_i]]
                ):
                    return False
            return True
    
        return  in_cone(verts, indices, ia, ib) and \
                in_cone(verts, indices, ib, ia) and \
                diagonalie(verts, indices, ia, ib)


# ============================================================================
# GEOMETRY OPERATOR CLASS
# ============================================================================

class GeometryOperator:
    """
    Geometry transformation and manipulation methods.
    All methods in this class perform operations and return modified geometry.
    """

    @staticmethod
    def reorder_vertices(face, is_upward):
        """Re-order vertices of a face to enforce winding in XY projection.
        
        This function ensures:
        1. Vertex order is normalized in XY projection
        2. The face starts from a consistent reference point (minimum sum of coordinates)
        3. Vertex direction matches the expected is_upward orientation
        
        Args:
            face: numpy array, shape=(n, 3), sequence of vertices of a face
            is_upward: bool, True for CCW from above, False for CW from above
        
        Returns:
            reordered_face: numpy array, shape=(n, 3), properly ordered vertices
        """
        face = np.asarray(face, dtype=float)
        if len(face) <= 1:
            return face

        # Rotate to a stable start vertex first (minimum x+y+z)
        min_index = np.argmin(np.sum(face, axis=1))
        face = np.roll(face, -min_index, axis=0)

        # Signed area in XY by shoelace: >0 => CCW, <0 => CW
        x = face[:, 0]
        y = face[:, 1]
        signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        is_ccw = signed_area > 0

        # is_upward=True => enforce CCW; is_upward=False => enforce CW
        if (is_upward and not is_ccw) or ((not is_upward) and is_ccw):
            face = face[::-1]

        # Re-align start vertex after potential reversal
        min_index = np.argmin(np.sum(face, axis=1))
        face = np.roll(face, -min_index, axis=0)

        return face

    @staticmethod
    def compute_max_inscribed_quadrilateral(vertices):
        """
        Compute the maximum area inscribed quadrilateral from a convex polygon.
        
        Parameters
        ----------
        vertices : list or numpy.ndarray
            List of vertices of a convex polygon, ordered counter-clockwise.
        
        Returns
        -------
        list
            List of four vertices forming the maximum area inscribed quadrilateral,
            or the original vertices if fewer than 4 vertices exist.
        """
        n = len(vertices)
        
        if n <= 4:
            return vertices
        
        max_area = 0.0
        best_quad_indices = None
        
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        quad_vertices = [vertices[i], vertices[j], vertices[k], vertices[l]]
                        area = GeometryBasic.polygon_area_2d(quad_vertices)
                        
                        if area > max_area:
                            max_area = area
                            best_quad_indices = [i, j, k, l]
        
        if best_quad_indices is not None:
            return [vertices[idx] for idx in best_quad_indices]
        else:
            return vertices

    @staticmethod
    def process_hole(hole, faces, check_projection=True):
        """
        Process a hole to determine if it should be skipped based on geometric conditions.
        
        Parameters
        ----------
        hole : numpy.ndarray
            A 2D array of shape (n, 3) representing the 3D coordinates of the hole polygon.
        faces : list of numpy.ndarray
            A list of 2D arrays, each representing a 3D face polygon with shape (m, 3).
        check_projection : bool, optional
            If True, checks whether the projection of the hole overlaps with projections of other faces.
            Default is True.
        
        Returns
        -------
        bool
            True if the hole should be skipped (i.e., it fully coincides with a face and meets projection criteria),
            False otherwise.
        """
        for other_face in faces:
            # 1. Check for complete 3D coincidence
            if GeometryValidator._is_same_polygon(hole, other_face):
                if not check_projection:
                    return True  # Skip directly
                
                # 2. Check if projection overlaps with other faces (excluding itself)
                has_projection_overlap = False
                for face in faces:
                    if GeometryValidator._is_same_polygon(hole, face):
                        continue  # Skip itself
                    
                    # Check if projections are the same (allowing reverse order)
                    if GeometryValidator._is_same_polygon(hole, face, projection=True):
                        has_projection_overlap = True
                        break
                
                # If no other face projection overlaps, skip this hole
                if not has_projection_overlap:
                    return True
        
        return False

    @staticmethod
    def merge_holes(verts_poly: np.ndarray, verts_holes: dict[int, np.ndarray]) -> np.ndarray:
        """
        Merge holes into a polygon by finding the shortest valid connection lines.
        
        Parameters
        ----------
        verts_poly : np.ndarray
            Array of vertices representing the outer polygon boundary.
        verts_holes : dict[int, np.ndarray]
            Dictionary mapping hole indices to their respective vertex arrays.
        
        Returns
        -------
        tuple[np.ndarray, list[np.ndarray]]
            A tuple containing merged vertices and merge lines.
        """
        if verts_holes is None or len(verts_holes) == 0:
            return verts_poly, []
        
        n_poly = len(verts_poly)
        indices_poly = list(range(n_poly))
        indices_holes = {}
        verts_all = verts_poly.copy()
        indices_all = indices_poly.copy()

        # Calculate vertex indices for all holes
        offset = n_poly
        for hole_id, verts_hole in verts_holes.items():
            n_hole = len(verts_hole)
            indices_holes[hole_id] = list(range(offset, offset + n_hole))
            verts_all = np.concatenate((verts_all, verts_hole))
            indices_all.extend(range(offset, offset + n_hole))
            offset += n_hole

        best_diagonals = {}

        # Iterate through each hole
        for hole_id, indices_hole in indices_holes.items():
            verts_hole = verts_holes[hole_id]
            n_hole = len(indices_hole)
            min_diagonal_length = float('inf')
            min_diagonal = None

            # Iterate through all vertices of current hole
            for hole_idx, hole_vert_idx in enumerate(indices_hole):
                hole_vertex = verts_hole[hole_idx]
                # Check connection with outer polygon vertices
                for poly_idx, poly_vertex_idx in enumerate(indices_poly):
                    poly_vertex = verts_poly[poly_idx]
                    okay = True
                    
                    # Check intersection with outer polygon edges
                    for poly_edge in range(n_poly):
                        poly_a = verts_poly[poly_edge]
                        poly_b = verts_poly[(poly_edge + 1) % n_poly]
                        if poly_idx in (poly_edge, (poly_edge + 1) % n_poly):
                            continue
                        if GeometryValidator._is_intersect(poly_vertex, hole_vertex, poly_a, poly_b):
                            okay = False
                            break
                    
                    if not okay:
                        continue
                        
                    # Check intersection with current hole edges
                    for hole_edge in range(n_hole):
                        hole_a = verts_hole[hole_edge]
                        hole_b = verts_hole[(hole_edge + 1) % n_hole]
                        if hole_idx in (hole_edge, (hole_edge + 1) % n_hole):
                            continue
                        if GeometryValidator._is_intersect(poly_vertex, hole_vertex, hole_a, hole_b):
                            okay = False
                            break
                    
                    if not okay:
                        continue
                        
                    # Check intersection with other holes
                    for other_id, other_indices in indices_holes.items():
                        if other_id == hole_id:
                            continue
                        other_verts = verts_holes[other_id]
                        for edge in range(len(other_verts)):
                            a = other_verts[edge]
                            b = other_verts[(edge + 1) % len(other_verts)]
                            if GeometryValidator._is_intersect(poly_vertex, hole_vertex, a, b):
                                okay = False
                                break
                        if not okay:
                            break
                    
                    if okay:
                        diagonal_length = np.linalg.norm(poly_vertex - hole_vertex)
                        if diagonal_length < min_diagonal_length:
                            min_diagonal_length = diagonal_length
                            min_diagonal = (poly_vertex_idx, hole_vert_idx)
            
            if min_diagonal is not None:
                best_diagonals[hole_id] = min_diagonal

        # Build connection line list
        diagonals = []
        for hole_id, (p_idx, h_idx) in best_diagonals.items():
            diagonals.append((p_idx, h_idx))
        diagonals = sorted(diagonals, key=lambda x: (x[0], -GeometryBasic.get_angle_tan(x[0], x[1], verts_all)))

        # Build new vertex list
        verts = []
        for idx in indices_poly:
            verts.append(verts_all[idx])
            
            # Check if there are connection lines starting from current vertex
            for diagonal in diagonals:
                if diagonal[0] == idx:
                    hole_vertex = diagonal[1]
                    # Find which hole this hole vertex belongs to
                    target_hole_id = None
                    target_hole_indices = None
                    for hole_id, hole_indices in indices_holes.items():
                        if hole_vertex in hole_indices:
                            target_hole_id = hole_id
                            target_hole_indices = hole_indices
                            break
                    
                    if target_hole_indices:
                        # Traverse hole vertices starting from connection endpoint
                        start_idx = target_hole_indices.index(hole_vertex)
                        n_hole = len(target_hole_indices)
                        # Add hole vertices in order
                        for i in range(n_hole + 1):
                            current_idx = target_hole_indices[(start_idx + i) % n_hole]
                            verts.append(verts_all[current_idx])
                        # Add current outer vertex again to close
                        verts.append(verts_all[idx])
        
        mergelines = [np.array([verts_all[pair[0]], verts_all[pair[1]]]) for pair in diagonals]

        return np.array(verts), mergelines

    @staticmethod
    def split_poly(verts: np.ndarray, indices: np.ndarray) -> Union[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Turn a simple polygon into a list of convex polygons.
        
        Parameters
        ----------
        verts : np.ndarray
            Array of 2D or 3D vertices
        indices : np.ndarray
            Array of vertex indices forming the polygon
            
        Returns
        -------
        tuple
            (list of convex polygon indices, list of split diagonals)
        """
        n = len(indices)
        i_concave = -1

        for ia in range(n):
            ia_prev, ia_next = (ia - 1) % n, (ia + 1) % n
            angle = GeometryBasic.angle(verts[indices[ia_prev]], verts[indices[ia]], verts[indices[ia_next]])
            if angle < 0:
                i_concave = ia
                break

        if i_concave == -1:
            return [indices], []

        i_break = -1
        min_diagonal_length = float('inf')
        for i in range(n):
            if i != i_concave and i != (i_concave+1) % n and i != (i_concave-1) % n:
                if GeometryValidator._is_diagonal(verts, indices, i_concave, i):
                    diagonal_length = np.linalg.norm(verts[indices[i_concave]] - verts[indices[i]])
                    if diagonal_length < min_diagonal_length:
                        i_break = i
                        min_diagonal_length = diagonal_length

        if i_break == -1:
            return [indices], []

        indices1 = []
        indices2 = []
        i_now = i_concave

        while i_now != i_break:
            indices1.append(indices[i_now])
            i_now = (i_now + 1) % n
        indices1.append(indices[i_break])

        while i_now != i_concave:
            indices2.append(indices[i_now])
            i_now = (i_now + 1) % n
        indices2.append(indices[i_concave])

        i1, diag1 = GeometryOperator.split_poly(verts, indices1)
        i2, diag2 = GeometryOperator.split_poly(verts, indices2)

        ret_diag = [[i_concave, i_break]]
        for diag in diag1:
            ret_diag.append(((diag[0] + i_concave) % n, (diag[1] + i_concave) % n))
        for diag in diag2:
            ret_diag.append(((diag[0] + i_break) % n, (diag[1] + i_break) % n))

        result_indices = i1 + i2

        return result_indices, ret_diag

    @staticmethod
    def create_airwalls(divide_lines):
        """
        Create quadrilateral faces from divide lines.
        
        Parameters
        ----------
        divide_lines : list
            List of line segments used in decomposition.
        
        Returns
        -------
        tuple
            (list of quad faces, list of quad normals)
        """
        line_groups = defaultdict(list)
        quad_faces = []
        quad_normals = []

        projected_lines = [line[:, :2] for line in divide_lines]
        z_lines = [(line[0, 2] + line[1, 2]) / 2 for line in divide_lines]

        # Group overlapping lines
        for i in range(len(projected_lines)):
            found_group = False
            for j in range(i):
                if (np.array_equal(projected_lines[i], projected_lines[j]) or 
                    np.array_equal(projected_lines[i], projected_lines[j][::-1])):
                    line_groups[j].append((divide_lines[i], z_lines[i]))
                    found_group = True
                    break
            if not found_group:
                line_groups[i].append((divide_lines[i], z_lines[i]))

        # Process each group to form quadrilaterals
        for group in line_groups.values():
            if len(group) < 2:
                continue

            group.sort(key=lambda x: x[1])

            for k in range(len(group) - 1):
                line1, _ = group[k]
                line2, _ = group[k + 1]

                quad = np.array([line1[0], line1[1], line2[1], line2[0]])
                quad_faces.append(quad)
                
                # Calculate normal
                v1 = quad[1] - quad[0]
                v2 = quad[3] - quad[0]
                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                quad_normals.append(normal)

        return quad_faces, quad_normals


# ============================================================================
# OBB UTILITIES
# ============================================================================

def create_obb(points, normal, min_scale=0.1):
    """
    Create an oriented bounding box (OBB) for a set of points.
    
    Parameters
    ----------
    points : np.ndarray
        Points, shape (N, 3)
    normal : np.ndarray
        Normal vector, shape (3,)
    min_scale : float
        Minimum allowed OBB scale
        
    Returns
    -------
    dict
        OBB parameters including center, scale, rotation
    """
    geometry = pygeos.multipoints(points)
    z_axis = np.array([0, 0, 1])
    z_r = normal
    
    if np.abs(z_r[0]) <= 1e-3 and np.abs(z_r[1]) <= 1e-3:
        z_r = z_axis
        min_rotated_rectangle = pygeos.minimum_rotated_rectangle(geometry)
        obb_coords = np.array(pygeos.get_coordinates(min_rotated_rectangle, include_z=True))[:-1]
        obb_coords = np.nan_to_num(obb_coords, nan=points[0, 2])
        obb_coords[:, 2] = (np.min(points[:, 2]) + np.max(points[:, 2])) / 2

        if len(obb_coords) <= 2:
            centroid = np.mean(points, axis=0)
            x_r, y_r = np.array([1, 0, 0]), np.array([0, 1, 0])
            rotation = np.array([x_r, y_r, z_r])
            rotation_matrix = R.from_matrix(rotation).as_matrix()
            l = max(np.ptp(points[:, 0]), min_scale)
            w = max(np.ptp(points[:, 1]), min_scale)
            h = max(np.ptp(points[:, 2]), min_scale)
            original_obb_centroid = centroid
        else:
            x_vec = obb_coords[1] - obb_coords[0]
            y_vec = obb_coords[3] - obb_coords[0]
            
            x_norm = np.linalg.norm(x_vec)
            y_norm = np.linalg.norm(y_vec)
            
            x_r = x_vec / x_norm if x_norm > 1e-6 else np.array([1, 0, 0])
            y_r = y_vec / y_norm if y_norm > 1e-6 else np.array([0, 1, 0])

            rotation = np.array([x_r, y_r, z_r])
            rotation_matrix = R.from_matrix(rotation).as_matrix()

            l = np.linalg.norm(obb_coords[1] - obb_coords[0])
            w = np.linalg.norm(obb_coords[3] - obb_coords[0])
            h = max(np.max(points[:, 2]) - np.min(points[:, 2]), min_scale)

            original_obb_centroid = np.mean(obb_coords, axis=0)
    else:
        x_r = np.cross(z_r, z_axis)
        y_r = np.cross(z_r, x_r)

        rotation = np.array([x_r, y_r, z_r])
        rotation_matrix = R.from_matrix(rotation).as_matrix()
        
        rotated_points = points.dot(rotation_matrix.T)

        l = max(np.ptp(rotated_points[:, 0]), min_scale)
        w = max(np.ptp(rotated_points[:, 1]), min_scale)
        h = max(np.ptp(rotated_points[:, 2]), min_scale)
        
        centroid = np.mean([
            [np.min(rotated_points[:, 0]), np.min(rotated_points[:, 1]), np.min(rotated_points[:, 2])],
            [np.max(rotated_points[:, 0]), np.max(rotated_points[:, 1]), np.max(rotated_points[:, 2])]
        ], axis=0)
        
        original_obb_centroid = np.dot(centroid, rotation_matrix)

    obb_params = {
        'center': original_obb_centroid,
        'scale': np.array([l, w, h]),
        'rotation': rotation_matrix,
    }

    return obb_params


def obb_to_face_vertices(obb_params):
    """
    Convert OBB parameters to face vertices (6 faces).
    
    Parameters
    ----------
    obb_params : dict
        OBB parameters with 'center', 'scale', 'rotation'
        
    Returns
    -------
    list
        List of 6 face vertices for the OBB box
    """
    center = obb_params['center']
    l, w, h = obb_params['scale']
    rotation_matrix = obb_params['rotation']

    # 8 corner points in local OBB coordinates
    offsets = np.array([
        [-l/2, -w/2, -h/2],
        [l/2, -w/2, -h/2],
        [l/2, w/2, -h/2],
        [-l/2, w/2, -h/2],
        [-l/2, -w/2, h/2],
        [l/2, -w/2, h/2],
        [l/2, w/2, h/2],
        [-l/2, w/2, h/2]
    ])

    # Rotate and translate
    corners = offsets.dot(rotation_matrix.T) + center

    # 6 faces of the box
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],  # bottom
        [corners[4], corners[7], corners[6], corners[5]],  # top
        [corners[0], corners[4], corners[5], corners[1]],  # front
        [corners[2], corners[6], corners[7], corners[3]],  # back
        [corners[0], corners[3], corners[7], corners[4]],  # left
        [corners[1], corners[5], corners[6], corners[2]],  # right
    ]

    return [np.array(face) for face in faces]

def calculate_wwr (cats, faces, normals):
    """
    Calculate Window-to-Wall Ratio (WWR) for a set of faces and their normals.
    
    Parameters
    ----------
    cats : list of str
        List of category IDs for each face
    faces : list of np.ndarray
        List of face vertices, each of shape (n, 3)
    normals : list of np.ndarray
        List of normal vectors corresponding to each face, each of shape (3,)
        
    Returns
    -------
    float
        The calculated WWR value
    """
    total_wall_area = 0.0
    total_window_area = 0.0
    
    for cat, face, normal in zip(cats, faces, normals):
        area = GeometryBasic.polygon_area_3d(face)
        if area < 1e-6:
            continue
        
        if abs(normal[2]) < 1e-3:
            if int(cat) == 1:  # Assuming category 1 represents windows
                total_window_area += area
            else:
                total_wall_area += area
            
    if total_wall_area + total_window_area == 0:
        return 0.0
    
    wwr = total_window_area / (total_wall_area + total_window_area)
    
    return wwr