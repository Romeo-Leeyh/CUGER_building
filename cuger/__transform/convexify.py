import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional
from collections import defaultdict

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
        """Check if line segment ab _is_intersects with line segment cd."""
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
    def _is_intersect(a, b, c, d):
        """
        Determine if two line segments intersect in 2D space.
        
        Parameters
        ----------
        a : array-like
            The first endpoint of the first line segment, as a 2D point (x, y).
        b : array-like
            The second endpoint of the first line segment, as a 2D point (x, y).
        c : array-like
            The first endpoint of the second line segment, as a 2D point (x, y).
        d : array-like
            The second endpoint of the second line segment, as a 2D point (x, y).
        
        Returns
        -------
        bool
            True if the two line segments intersect, False otherwise.
        """

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
        """Re-order vertices of a face to make the normal face upward or downward.
        
        Args:
            face: numpy array, shape=(n, 3), sequence of vertices of a face
            is_upward: bool，True for upward, False for downward
        
        Returns:
            reordered_face: numpy array, shape=(n, 3)
        """
        # calculate the sum of x, y, z of each vertex
        sum_xyz = np.sum(face, axis=1)
        min_index = np.argmin(sum_xyz)
        
        # reorder the vertices based on the minimum index
        if is_upward:
            face = np.roll(face, -min_index, axis=0)
        else:
            face = np.roll(face, -min_index + face.shape[0] - 1, axis=0)[::-1]
            
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
        """
        Logic to process a hole and determine if it should be skipped.

        Conditions:
        1. Hole completely coincides with a face in 3D
        2. If check_projection is True, additionally check if projection does not overlap with other faces

        Args:
            hole: numpy array, shape (n, 3)
            faces: list[numpy array], all face data
            check_projection: whether to check projection overlap

        Returns:
            bool: whether to skip this hole
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
            Dictionary mapping hole indices to their respective vertex arrays; each key is an integer 
            identifying a hole, and the corresponding value is a NumPy array of its vertices.
        
        Returns
        -------
        tuple[np.ndarray, list[np.ndarray]]
            A tuple containing:
            - indices_all: Array of merged vertices including both polygon and hole vertices in traversal order.
            - _is_diagonals: List of connection line segments represented as arrays of two points, 
              each connecting a polygon vertex to a hole vertex.
        """
        """
        Find the shortest valid connection line for each hole.
        
        Args:
            verts_poly: outer polygon vertex coordinates
            verts_holes: dictionary of hole vertex coordinates, key is hole index
        
        Returns:
            tuple: (indices_all, _is_diagonals)
                - indices_all: list of all vertex indices
                - _is_diagonals: list of connection lines, each element is (hole_vertex_idx, poly_vertex_idx)
        """
        if verts_holes is None or len(verts_holes) == 0:
            return verts_poly, []
        
        n_poly = len(verts_poly)  # Number of outer polygon vertices
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

        best__is_diagonals = {}  # Store the best connection for each hole

        # Iterate through each hole
        for hole_id, indices_hole in indices_holes.items():
            verts_hole = verts_holes[hole_id]
            n_hole = len(indices_hole)
            min__is_diagonal_length = float('inf')
            min__is_diagonal = None

            # Iterate through all vertices of current hole
            for hole_idx, hole_vert_idx in enumerate(indices_hole):
                hole_vertex = verts_hole[hole_idx]  # Use hole_idx instead of hole_vert_idx
                # Check connection with outer polygon vertices
                for poly_idx, poly_vertex_idx in enumerate(indices_poly):
                    poly_vertex = verts_poly[poly_idx]
                    okay = True
                    
                    # Check _is_intersection with outer polygon edges
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
                        
                    # Check _is_intersection with current hole edges
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
                        
                    # Check _is_intersection with other holes
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
                        # Calculate _is_diagonal length
                        _is_diagonal_length = np.linalg.norm(poly_vertex - hole_vertex)
                        if _is_diagonal_length < min__is_diagonal_length:
                            min__is_diagonal_length = _is_diagonal_length
                            min__is_diagonal = (poly_vertex_idx, hole_vert_idx)
            
            if min__is_diagonal is not None:
                best__is_diagonals[hole_id] = min__is_diagonal

        # Build connection line list
        _is_diagonals = []
        
        for hole_id, (p_idx, h_idx) in best__is_diagonals.items():
            _is_diagonals.append((p_idx, h_idx))
        _is_diagonals = sorted(_is_diagonals, key=lambda x: (x[0], -GeometryBasic.get_angle_tan(x[0], x[1], verts_all)))

        # Build new vertex list
        verts = []
        for idx in indices_poly:
            verts.append(verts_all[idx])
            
            # Check if there are connection lines starting from current vertex
            for _is_diagonal in _is_diagonals:
                if _is_diagonal[0] == idx:
                    hole_vertex = _is_diagonal[1]
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
                        for i in range(n_hole + 1):  # +1 is to return to the starting point
                            current_idx = target_hole_indices[(start_idx + i) % n_hole]
                            verts.append(verts_all[current_idx])
                        # Add current outer vertex again to close
                        verts.append(verts_all[idx])
        
        mergelines = [np.array([verts_all[pair[0]], verts_all[pair[1]]]) for pair in _is_diagonals]

        return np.array(verts), mergelines
    
    @staticmethod
    def split_poly(verts: np.ndarray, indices: np.ndarray) -> Union[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Turn a simple polygon into a list of convex polygons that shares the same area.
        This divide-and-conquer methods base on Arkin, Ronald C.'s report (1987).
        "Path planning for a vision-based autonomous robot"

        :param verts:       np.ndarray (#verts, 2)  a list of 2D-vertices position
        :param indices:     np.ndarray (#vert, )    a list of polygon vertex index (to array `verts`)
        :return:  ([np.ndarray], [(int, int)])
            a list of indices of `verts` that constructs convex areas
            e.g: [np.array(p1_i1, p1_i2, p1_i3, ..), np.array(p2_i1, ...), ..]

            list of _is_diagonals that splits the input polygon.
            e.g: [(diag1_a_index, diag1_b_index), ...]
        """

        # find concave vertex
        n = len(indices)
        i_concave = -1

        for ia in range(n):
            ia_prev, ia_next = (ia - 1) % n, (ia + 1) % n
            angle = GeometryBasic.angle(verts[indices[ia_prev]], verts[indices[ia]], verts[indices[ia_next]])
            
            if angle < 0:
                i_concave = ia
                break

        # if there is no concave vertex, which means current polygon is convex. Return itself directly
        if i_concave == -1:
            return [indices], []

        # Find vertex i_break that `<i_concave, i_break>` is an internal edge
        i_break = -1
        min__is_diagonal_length = float('inf')  # initialize with infinity
        for i in range(n):
            if i != i_concave and i != (i_concave+1) % n and i != (i_concave-1) % n:
                if GeometryValidator._is_diagonal(verts, indices, i_concave, i):
                    # Calculate the length of the _is_diagonal
                    _is_diagonal_length = np.linalg.norm(verts[indices[i_concave]] - verts[indices[i]])

                    # Update i_break if the current _is_diagonal is shorter
                    if _is_diagonal_length < min__is_diagonal_length:
                        i_break = i
                        min__is_diagonal_length = _is_diagonal_length

        # Not find (should not happen!)
        if i_break == -1:
            # Just keep that weird region for now
            # TBD: raise a warning
            return [indices], []

        # Split the simple polygon by <i_concave, i_break>
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

        # keep convexifying new-ly generated two areas in a recursive manner
        i1, diag1 = GeometryOperator.split_poly(verts, indices1)
        i2, diag2 = GeometryOperator.split_poly(verts, indices2)

        # merge results from recursively convexify
        ret_diag = [[i_concave, i_break]]
        for diag in diag1:
            ret_diag.append(((diag[0] + i_concave) % n, (diag[1] + i_concave) % n))
        for diag in diag2:
            ret_diag.append(((diag[0] + i_break) % n, (diag[1] + i_break) % n))

        result_indices = i1 + i2
        """
        for new_indices in [indices1, indices2]:
            count = 0
            for ia in range(len(new_indices)):
                ia_prev, ia_next = (ia - 1) % n, (ia + 1) % n
                angle = BasicOptions.angle(verts[indices[ia_prev]], verts[indices[ia]], verts[indices[ia_next]])
                if angle > 0:
                    count += 1
            
            if count > 4:    
                return GeometryOperator.split_poly(verts, new_indices)
        """

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

        # Get the projection coordinates and midpoint heights
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
                continue  # Skip groups with fewer than 2 lines

            # Sort lines in the group by midpoint height
            group.sort(key=lambda x: x[1])  # Sort by z value

            # Create quadrilaterals
            for k in range(len(group) - 1):
                line1, _ = group[k]
                line2, _ = group[k + 1]

                # Skip if the two lines are geometrically identical (same endpoints)
                if (np.allclose(line1, line2) or np.allclose(line1, line2[::-1])):
                    continue

                quad_face = np.array([line1[0], line1[1], line2[1], line2[0]])
                normal_vector = np.cross(line1[0] - line1[1], line1[0] - line2[0])
                quad_normal = normal_vector / np.linalg.norm(normal_vector)

                quad_faces.append(quad_face)
                quad_normals.append(quad_normal)

        return quad_faces, quad_normals

# ============================================================================
# MAIN CONVEXIFY 
# ============================================================================

@staticmethod
def convexify_faces(cat, idd, normal, faces, holes, 
                    valid_face=True, clean_quad=False):
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
    valid_face : bool, optional
        If True, filter out invalid faces using GeometryValidator._is_valid_face().
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
        is_upward = normal[idx][2] > 0
        faces[idx] = GeometryOperator.reorder_vertices(faces[idx], is_upward=is_upward)
        if holes[idx]:
            for i in range(len(holes[idx])):
                holes[idx][i] = GeometryOperator.reorder_vertices(holes[idx][i], is_upward=is_upward)
    
    print("--Faces reordering done--")

    for idx, face in enumerate(faces):
        # Skip invalid faces if validation is enabled
        if valid_face and not GeometryValidator._is_valid_face(face):
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
            
            subfaces = []
            for poly in polys:
                if valid_face: 
                    if GeometryValidator._is_valid_face(verts[poly]):
                        subfaces.append(verts[poly])
                    if not GeometryValidator._is_valid_face(verts[poly]):
                        print(f"Skipping invalid sub-face in face {idd[idx]}")
                        continue
                if clean_quad and len(poly) > 4:
                    quad_poly = GeometryOperator.compute_max_inscribed_quadrilateral(verts[poly])
                    if valid_face and not GeometryValidator._is_valid_face(quad_poly):
                        print(f"Skipping invalid quadrilateral sub-face in face {idd[idx]}")
                        continue
                    subfaces.append(np.array(quad_poly))
                else:
                    subfaces.append(verts[poly])

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
            convex_cat.append(cat[idx])
            convex_idd.append(idd[idx])
            convex_normal.append(normal[idx])
            convex_faces.append(face)
    
    print ("--Faces splitting done--")

    # Create quadrilateral air walls
    quad_faces, quad_normals = GeometryOperator.create_airwalls(divide_lines)

    for i, face in enumerate(quad_faces):
        convex_cat.append("2")
        convex_idd.append(f"a_{i}")
        convex_normal.append(quad_normals[i])
        convex_faces.append(face)

    return convex_cat, convex_idd, convex_normal, convex_faces, divide_lines

def plot_faces(faces, lines, file_path, _fig_show =False):
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=45, azim=15)

    for face in faces:
        x, y, z = face[:, 0], face[:, 1], face[:, 2]

        x = np.append(x, x[0])
        y = np.append(y, y[0])
        z = np.append(z, z[0])    

        ax.plot(x, y, z, 'purple')  
        ax.scatter(x, y, z, c='black', marker='o', s=20)
        
    if lines:
        for line in lines:
            x, y, z = line[:, 0], line[:, 1], line[:, 2] 

            ax.plot(x, y, z, 'blue')  

    all_points = np.vstack(faces)  
    if lines:
        all_points = np.vstack([all_points] + lines) 

    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    z_min, z_max = np.min(all_points[:, 2]), np.max(all_points[:, 2])

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    mid_x = (x_max + x_min) / 2.0
    mid_y = (y_max + y_min) / 2.0
    mid_z = (z_max + z_min) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_box_aspect([1, 1, 1])
    
    plt.axis('off')
    ax.set_axis_off()
    if _fig_show:
        plt.show()
    plt.savefig(file_path, dpi=300)
    plt.close()
