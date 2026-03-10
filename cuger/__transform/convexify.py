"""
    This module implements the convexification process for polygonal faces in a building model. The main function, `convexify_faces`, takes in a batch of faces and their associated holes, performs geometric normalization, merges holes into the faces, and then applies a convex decomposition algorithm to generate convex sub-faces. Additionally, it creates quadrilateral air-wall patches along the divide lines generated during the convexification process. The function returns the convexified faces and the corresponding air-wall patches.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import geometry classes from geometry module
from .geometry import GeometryValidator, GeometryOperator

# ============================================================================
# MAIN CONVEXIFY 
# ============================================================================

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
    
    # Reorder non-wall faces in top view:
    # - outer face: counter-clockwise
    # - holes: clockwise
    for idx in range(len(faces)):
        if np.abs(normal[idx][2]) > 1e-3:
            faces[idx] = GeometryOperator.reorder_vertices(faces[idx], is_upward=True)
            if holes[idx]:
                for i in range(len(holes[idx])):
                    holes[idx][i] = GeometryOperator.reorder_vertices(holes[idx][i], is_upward=False)
    
    print("--Faces reordering done--")

    for idx, face in enumerate(faces):
        # Skip invalid faces if validation is enabled
        if valid_face and not GeometryValidator._is_valid_face(face):
            print(f"    Skipping invalid face {idd[idx]}")
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
                        print(f"    Skipping invalid sub-face in face {idd[idx]}")
                        continue
                if clean_quad and len(poly) > 4:
                    quad_poly = GeometryOperator.compute_max_inscribed_quadrilateral(verts[poly])
                    if valid_face and not GeometryValidator._is_valid_face(quad_poly):
                        print(f"    Skipping invalid quadrilateral sub-face in face {idd[idx]}")
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
