import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional
from collections import defaultdict

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ============================================================================
# GEOMETRY VALIDATOR CLASS - 几何判断类
# ============================================================================

class GeometryValidator:
    """
    Geometry validation and checking methods.
    All methods in this class return boolean values or perform geometric tests.
    """
    
    @staticmethod
    def left_on(p1, p2, p3):
        """Check if p3 is on the left side of line p1->p2 in 2D."""
        p1_2d = p1[:2]
        p2_2d = p2[:2]
        p3_2d = p3[:2]
        cross = np.cross(p2_2d - p1_2d, p3_2d - p2_2d)
        if cross > 0 and np.abs(cross) < 1e-6 * np.linalg.norm(p2_2d - p1_2d) * np.linalg.norm(p3_2d - p2_2d):
            return False
        return cross > 0
    
    @staticmethod
    def collinear(p1, p2, p3):
        """Check if three points are collinear."""
        area = np.cross(p2 - p1, p3 - p2)
        dist = area / (np.dot(p1, p2 - p1) + 1e-6)
        return np.abs(dist) < 1e-3

    @staticmethod
    def between(p1, p2, p3):
        """Check if p3 is between p1 and p2."""
        p1_2d = p1[:2]
        p2_2d = p2[:2]
        p3_2d = p3[:2]
        if p1_2d[0] != p2_2d[0]:
            return (p1_2d[0] <= p3_2d[0] <= p2_2d[0]) or (p1_2d[0] >= p3_2d[0] >= p2_2d[0])
        else:
            return (p1_2d[1] <= p3_2d[1] <= p2_2d[1]) or (p1_2d[1] >= p3_2d[1] >= p2_2d[1])

    @staticmethod
    def intersect(a, b, c, d):
        """Check if line segment ab intersects with line segment cd."""
        a_2d = a[:2]
        b_2d = b[:2]
        c_2d = c[:2]
        d_2d = d[:2]
        if GeometryValidator.collinear(a_2d, b_2d, c_2d):
            return GeometryValidator.between(a_2d, b_2d, c_2d)
        if GeometryValidator.collinear(a_2d, b_2d, d_2d):
            return GeometryValidator.between(a_2d, b_2d, d_2d)
        if GeometryValidator.collinear(c_2d, d_2d, a_2d):
            return GeometryValidator.between(c_2d, d_2d, a_2d)
        if GeometryValidator.collinear(c_2d, d_2d, b_2d):
            return GeometryValidator.between(c_2d, d_2d, b_2d)
        cd_cross = np.logical_xor(GeometryValidator.left_on(a_2d, b_2d, c_2d), GeometryValidator.left_on(a_2d, b_2d, d_2d))
        ab_cross = np.logical_xor(GeometryValidator.left_on(c_2d, d_2d, a_2d), GeometryValidator.left_on(c_2d, d_2d, b_2d))
        return ab_cross and cd_cross
    
    @staticmethod
    def is_obtuse(v1, v2, v3):
        """Check if angle at v2 is obtuse."""
        return GeometryValidator.angle(v1, v2, v3) > 90
    
    @staticmethod
    def is_valid_face(vertices, area_eps=1e-8):
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
    def is_same_polygon(polygon1, polygon2, projection=False):
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
    def diagonal(verts: np.ndarray, indices: np.ndarray, ia: int, ib: int) -> bool:
        """Check if diagonal between two vertices is valid."""
        def in_cone(verts: np.ndarray, indices: np.ndarray, ia: int, ib: int) -> bool:
            n = len(indices)
            ia_prev = ia - 1 if ia - 1 >= 0 else n - 1
            ia_next = ia + 1 if ia + 1 < n else 0

            ia, ib = indices[ia], indices[ib]
            ia_prev, ia_next = indices[ia_prev], indices[ia_next]

            # Convex
            if GeometryValidator.left_on(verts[ia_prev], verts[ia], verts[ia_next]):
                return GeometryValidator.left_on(verts[ia], verts[ib], verts[ia_prev]) and \
                    GeometryValidator.left_on(verts[ib], verts[ia], verts[ia_next])
            # Concave
            return not (GeometryValidator.left_on(verts[ia], verts[ib], verts[ia_next]) and \
                        GeometryValidator.left_on(verts[ib], verts[ia], verts[ia_prev]))

        def diagonalie(verts: np.ndarray, indices: np.ndarray, ia: int, ib: int) -> bool:
            n = len(indices)
            for now_i in range(n):
                if indices[now_i] == indices[ia] or indices[now_i] == indices[ib]:
                    continue
                next_i = (now_i + 1) % n
                if indices[next_i] == indices[ia] or indices[next_i] == indices[ib]:
                    continue

                if GeometryValidator.intersect(
                        verts[indices[ia]], verts[indices[ib]],
                        verts[indices[now_i]], verts[indices[next_i]]
                ):
                    return False
            return True
    
        return  in_cone(verts, indices, ia, ib) and \
                in_cone(verts, indices, ib, ia) and \
                diagonalie(verts, indices, ia, ib)
    
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


# ============================================================================
# GEOMETRY OPERATOR CLASS - 几何操作类
# ============================================================================

class GeometryOperator:
    """
    Geometry transformation and manipulation methods.
    All methods in this class perform operations and return modified geometry.
    """
    
    @staticmethod
    def reorder_vertices(face, is_upward=None):
        """
        Re-order vertices of a face to ensure counter-clockwise order in top view (XY plane projection).
        
        This function forces all vertices to be arranged in counter-clockwise order when viewed from above,
        regardless of the face normal direction.
        
        Parameters
        ----------
        face : numpy.ndarray
            Array of shape (n, 3) representing the sequence of vertices of a face.
        is_upward : bool, optional
            Kept for backward compatibility but no longer affects the ordering logic.
            All faces are now forced to counter-clockwise order in top view.
        
        Returns
        -------
        reordered_face : numpy.ndarray
            Array of shape (n, 3) with vertices reordered to counter-clockwise in top view.
        """
        # Project vertices to XY plane (top view) by ignoring Z coordinate
        vertices_2d = face[:, :2]
        
        # Calculate signed area using the shoelace formula
        n = len(vertices_2d)
        signed_area = 0.0
        for i in range(n):
            x1, y1 = vertices_2d[i]
            x2, y2 = vertices_2d[(i + 1) % n]
            signed_area += (x1 * y2 - x2 * y1)
        
        # If signed area is negative, vertices are in clockwise order
        # Reverse the order to make them counter-clockwise
        if signed_area < 0:
            face = face[::-1]
        
        return face
    
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
                        area = GeometryOperator.polygon_area_2d(quad_vertices)
                        
                        if area > max_area:
                            max_area = area
                            best_quad_indices = [i, j, k, l]
        
        if best_quad_indices is not None:
            return [vertices[idx] for idx in best_quad_indices]
        else:
            return vertices
    
    @staticmethod
    def get_angle_tan(p1, p2, verts_all):
        """Get angle tangent between two vertex indices."""
        vec = verts_all[p2] - verts_all[p1]
        return np.arctan2(vec[1], vec[0])
    
    @staticmethod
    def process_hole(hole, faces, check_projection=True):
        """
        Check if a hole should be skipped based on projection overlap with existing faces.
        
        Parameters
        ----------
        hole : np.ndarray
            Hole vertices to check.
        faces : list
            List of existing face vertices.
        check_projection : bool
            Whether to perform projection-based checking.
        
        Returns
        -------
        bool
            True if hole should be skipped, False otherwise.
        """
        if not check_projection:
            return False
        
        for face in faces:
            if GeometryValidator.is_same_polygon(hole, face, projection=True):
                return True
        return False
    
    @staticmethod
    def merge_holes(verts_poly: np.ndarray, verts_holes: dict[int, np.ndarray]) -> tuple:
        """
        Merge holes into outer polygon boundary.
        
        Parameters
        ----------
        verts_poly : np.ndarray
            Array of vertices representing the outer polygon boundary.
        verts_holes : dict[int, np.ndarray]
            Dictionary mapping hole indices to their respective vertex arrays.
        
        Returns
        -------
        tuple[np.ndarray, list[np.ndarray]]
            A tuple containing:
            - indices_all: Array of merged vertices.
            - diagonals: List of connection line segments.
        """
        if not verts_holes:
            return verts_poly, []
        
        n_poly = len(verts_poly)
        indices_poly = list(range(n_poly))
        indices_holes = {}
        
        vertex_offset = n_poly
        for hole_id, hole_verts in verts_holes.items():
            n_hole = len(hole_verts)
            indices_holes[hole_id] = list(range(vertex_offset, vertex_offset + n_hole))
            vertex_offset += n_hole
        
        verts_all = np.vstack([verts_poly] + [verts_holes[hid] for hid in sorted(verts_holes.keys())])
        
        diagonals = []
        
        for hole_id in sorted(verts_holes.keys()):
            indices_hole = indices_holes[hole_id]
            verts_hole = verts_holes[hole_id]
            
            min_distance = float('inf')
            min_diagonal = None

            for hole_idx, hole_vert_idx in enumerate(indices_hole):
                hole_vertex = verts_hole[hole_idx]
                for poly_idx, poly_vertex_idx in enumerate(indices_poly):
                    poly_vertex = verts_poly[poly_idx]
                    
                    distance = np.linalg.norm(hole_vertex - poly_vertex)
                    
                    if distance < min_distance:
                        min_distance = distance
                        min_diagonal = (poly_vertex_idx, hole_vert_idx, poly_idx, hole_idx)
            
            if min_diagonal:
                poly_vertex_idx, hole_vertex_idx, poly_insert_idx, hole_start_idx = min_diagonal
                
                poly_vertex = verts_all[poly_vertex_idx]
                hole_vertex = verts_all[hole_vertex_idx]
                
                diagonals.append(np.array([poly_vertex, hole_vertex]))
                
                target_hole_indices = indices_holes.get(hole_id, [])
                
                if target_hole_indices:
                    start_idx = target_hole_indices.index(hole_vertex_idx)
                    n_hole = len(target_hole_indices)
                    
                    new_segment = []
                    for i in range(n_hole + 1):
                        current_idx = target_hole_indices[(start_idx + i) % n_hole]
                        new_segment.append(current_idx)
                    
                    indices_poly = (
                        indices_poly[:poly_insert_idx + 1] + 
                        new_segment + 
                        indices_poly[poly_insert_idx:]
                    )
        
        merged_verts = verts_all[indices_poly]
        return merged_verts, diagonals
    
    @staticmethod
    def split_poly(verts: np.ndarray, indices: np.ndarray) -> Union[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Split a polygon into convex sub-polygons.
        
        Parameters
        ----------
        verts : np.ndarray
            Array of vertex coordinates.
        indices : np.ndarray
            Array of indices defining the polygon.
        
        Returns
        -------
        tuple
            (list of convex polygons, list of diagonal pairs)
        """
        n = len(indices)
        
        if n < 3:
            return [], []
        
        if n == 3:
            return [indices], []
        
        # Try to find a valid diagonal
        for i in range(n):
            for j in range(i + 2, n):
                if (j == (i + n - 1) % n):
                    continue
                
                if GeometryValidator.diagonal(verts, indices, i, j):
                    # Split into two parts
                    poly1_indices = indices[i:j+1]
                    poly2_indices = np.concatenate([indices[j:], indices[:i+1]])
                    
                    # Recursively split
                    polys1, diags1 = GeometryOperator.split_poly(verts, poly1_indices)
                    polys2, diags2 = GeometryOperator.split_poly(verts, poly2_indices)
                    
                    return polys1 + polys2, diags1 + diags2 + [(indices[i], indices[j])]
        
        # Already convex or cannot split
        return [indices], []
    
    @staticmethod
    def split_quad(verts: np.ndarray, indices: np.ndarray) -> Union[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Split a convex polygon into triangles or quadrilaterals without obtuse angles.
        
        Parameters
        ----------
        verts : np.ndarray
            Array of 2D vertex positions.
        indices : np.ndarray
            Array of indices referring to vertices.
        
        Returns
        -------
        list
            List of sub-polygon indices (triangles or quadrilaterals).
        """
        n = len(indices)
        
        if n < 3:
            return []
        
        if n == 3:
            return [indices]
        
        if n == 4:
            return [indices]
        
        polygons = []
        
        for i in range(n):
            if i < n - 2:
                if not GeometryValidator.is_obtuse(verts[indices[i]], verts[indices[i+1]], verts[indices[i+2]]):
                    polygons.append(indices[[i, i+1, i+2]])
                else:
                    if i < n - 3:
                        polygons.append(indices[[i, i+1, i+2, i+3]])
        
        return polygons


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================

class BasicOptions(GeometryValidator):
    """Backward compatibility: BasicOptions -> GeometryValidator"""
    pass


class Geometry_Option:
    """Backward compatibility: Geometry_Option with mixed methods"""
    
    # Validator methods
    is_valid_face = staticmethod(GeometryValidator.is_valid_face)
    is_same_polygon = staticmethod(GeometryValidator.is_same_polygon)
    
    # Operator methods
    reorder_vertices = staticmethod(GeometryOperator.reorder_vertices)
    compute_max_inscribed_quadrilateral = staticmethod(GeometryOperator.compute_max_inscribed_quadrilateral)
    process_hole = staticmethod(GeometryOperator.process_hole)
    merge_holes = staticmethod(GeometryOperator.merge_holes)
    split_poly = staticmethod(GeometryOperator.split_poly)
    split_quad = staticmethod(GeometryOperator.split_quad)


# ============================================================================
# MAIN CONVEXIFY CLASS
# ============================================================================

class MoosasConvexify:
    """Main class for convexification operations."""
    
    @staticmethod
    def create_quadrilaterals(divide_lines):
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
        if not divide_lines:
            return [], []
        
        quad_faces = []
        quad_normals = []
        
        line_dict = defaultdict(list)
        
        for line in divide_lines:
            if len(line) != 2:
                continue
            
            p1, p2 = line[0], line[1]
            key1 = tuple(p1)
            key2 = tuple(p2)
            
            line_dict[key1].append(p2)
            line_dict[key2].append(p1)
        
        used_pairs = set()
        
        for key, neighbors in line_dict.items():
            if len(neighbors) >= 2:
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        p1 = np.array(key)
                        p2 = neighbors[i]
                        p3 = neighbors[j]
                        
                        pair_key = tuple(sorted([tuple(p2), tuple(p3)]))
                        
                        if pair_key in used_pairs:
                            continue
                        
                        # Check if p3 is in the neighbors of p2
                        p2_tuple = tuple(p2)
                        p3_tuple = tuple(p3)
                        if p2_tuple in line_dict:
                            # Check if any neighbor of p2 matches p3
                            p3_found = False
                            for neighbor in line_dict[p2_tuple]:
                                if np.allclose(neighbor, p3):
                                    p3_found = True
                                    break
                            if p3_found:
                                quad = np.array([p1, p2, p3, p1])
                                
                                v1 = p2 - p1
                                v2 = p3 - p1
                                normal = np.cross(v1, v2)
                                
                                if np.linalg.norm(normal) > 1e-6:
                                    normal = normal / np.linalg.norm(normal)
                                    quad_faces.append(quad)
                                    quad_normals.append(normal)
                                    used_pairs.add(pair_key)
        
        return quad_faces, quad_normals
    
    @staticmethod
    def convexify_faces(cat, idd, normal, faces, holes, 
                       is_valid_face=False, clean_quad=False):
        """
        Convexify polygonal faces and generate quadrilateral air-wall patches.

        This function processes a batch of polygonal faces (possibly with holes) 
        and performs geometric normalization, hole integration, convex decomposition, 
        and quadrilateral generation.

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
        is_valid_face : bool, optional
            If True, filter out invalid faces using GeometryValidator.is_valid_face().
            Default is False.
        clean_quad : bool, optional
            If True, apply quadrilateralization to newly generated convex faces.
            Default is False.

        Returns
        -------
        convex_cat : list[str]
            Categories of all resulting faces.
        convex_idd : list[str]
            IDs of resulting faces.
        convex_normal : list[array-like]
            Normals of resulting faces.
        convex_faces : list[list[array-like]]
            Vertex lists of resulting faces.
        divide_lines : list[array-like]
            All generated split/merge lines.
        """
        convex_cat = []
        convex_idd = []
        convex_normal = []
        convex_faces = []
        divide_lines = []
        
        # Face reordering to counter-clockwise in top view
        for idx in range(len(faces)):
            faces[idx] = GeometryOperator.reorder_vertices(faces[idx])
            if holes[idx]:
                for i in range(len(holes[idx])):
                    holes[idx][i] = GeometryOperator.reorder_vertices(holes[idx][i])
        
        print("--Faces reordering done--")

        for idx, face in enumerate(faces):
            # Always check for severely degenerate faces to prevent errors
            if len(face) < 3:
                print(f"Skipping face {idd[idx]} with < 3 vertices")
                continue
            
            # Skip invalid faces if validation is enabled
            if is_valid_face and not GeometryValidator.is_valid_face(face):
                print(f"Skipping invalid face {idd[idx]}")
                continue
            
            if np.abs(normal[idx][2]) > 1e-3:  # Not wall
                poly_ex = face
                
                # Hole Merging
                poly_in = {}
                if holes[idx]:
                    for i in range(len(holes[idx])):
                        hole = holes[idx][i]
                        should_skip = GeometryOperator.process_hole(hole, faces, check_projection=True)
                        if should_skip:
                            continue
                        poly_in[i] = hole
   
                    verts, mergelines = GeometryOperator.merge_holes(poly_ex, poly_in)
                    
                    if mergelines:
                        divide_lines.extend(mergelines)
                        
                else:
                    verts = poly_ex

                # Convexification
                indices = list(range(len(verts)))
                polys, diags = GeometryOperator.split_poly(verts, indices)
                
                subfaces = [verts[poly] for poly in polys]
                
                # Apply quadrilateralization if enabled
                if clean_quad and len(subfaces) > 1:
                    processed_subfaces = []
                    for subface in subfaces:
                        if len(subface) > 4:
                            quad_face = GeometryOperator.compute_max_inscribed_quadrilateral(subface)
                            processed_subfaces.append(quad_face)
                        else:
                            processed_subfaces.append(subface)
                    subfaces = processed_subfaces
                
                if len(subfaces) == 1:
                    for i, subface in enumerate(subfaces):
                        convex_cat.append(cat[idx])
                        convex_idd.append(idd[idx])
                        convex_normal.append(normal[idx])
                        convex_faces.append(subface)
                else:
                    for i, subface in enumerate(subfaces):
                        convex_cat.append(cat[idx])
                        convex_idd.append(f"#{idd[idx]}_{i}")
                        convex_normal.append(normal[idx])
                        convex_faces.append(subface)
                    
                    if diags:
                        sublines = [np.array([verts[pair[0]], verts[pair[1]]]) for pair in diags]
                        divide_lines.extend(sublines)

            else:
                # Wall face - keep as is
                convex_cat.append(cat[idx])
                convex_idd.append(idd[idx])
                convex_normal.append(normal[idx])
                convex_faces.append(face)
        
        print("--Faces splitting done--")

        # Create quadrilateral air walls
        quad_faces, quad_normals = MoosasConvexify.create_quadrilaterals(divide_lines)
        
        for i, face in enumerate(quad_faces):
            convex_cat.append("2")
            convex_idd.append(f"a_{i}")
            convex_normal.append(quad_normals[i])
            convex_faces.append(face)

        return convex_cat, convex_idd, convex_normal, convex_faces, divide_lines

    @staticmethod
    def plot_faces(faces, lines, file_path, _fig_show=False):
        """
        Plot faces and lines for visualization.
        
        Parameters
        ----------
        faces : list
            List of face vertices.
        lines : list
            List of line segments.
        file_path : str
            Output file path.
        _fig_show : bool
            Whether to show the figure.
        """
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        for face in faces:
            face_array = np.array(face)
            xs = face_array[:, 0]
            ys = face_array[:, 1]
            zs = face_array[:, 2]
            ax.plot(xs, ys, zs, 'b-', linewidth=0.5)
        
        for line in lines:
            line_array = np.array(line)
            ax.plot(line_array[:, 0], line_array[:, 1], line_array[:, 2], 'r-', linewidth=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.savefig(file_path, dpi=150)
        
        if _fig_show:
            plt.show()
        else:
            plt.close()
