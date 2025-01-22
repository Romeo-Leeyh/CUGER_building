import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple
from collections import defaultdict

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class BasicOptions:
    @staticmethod
    def left_on(p1, p2, p3):
        # 取xy平面投影
        p1_2d = p1[:2]
        p2_2d = p2[:2]
        p3_2d = p3[:2]
        cross = np.cross(p2_2d - p1_2d, p3_2d - p2_2d)
        if cross > 0 and np.abs(cross) < 1e-6 * np.linalg.norm(p2_2d - p1_2d) * np.linalg.norm(p3_2d - p2_2d):
            return False
        return cross > 0
    
    @staticmethod
    def angle(p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p2
        cross = np.cross(v1, v2)
        dot = np.dot(v1, v2)
        
        # 处理近似平行的情况
        if np.linalg.norm(cross) < 1e-3 * np.linalg.norm(v1) * np.linalg.norm(v2):
            return 0  # 平行情况
        
        # 计算角度
        angle_rad = np.arccos(np.clip(dot/(np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        # 根据叉积z分量判断旋转方向
        return angle_deg if cross[2] > 0 else -angle_deg

    @staticmethod
    def get_angle_tan(p1, p2, verts_all):
        """计算两点连线与x轴正方向的夹角(弧度制)"""
        vec = verts_all[p2] - verts_all[p1]
        return np.arctan2(vec[1], vec[0])
    
    @staticmethod
    def is_obtuse(v1, v2, v3):
        return BasicOptions.angle(v1, v2, v3) > 90

    @staticmethod
    def collinear(p1, p2, p3):
        area = np.cross(p2 - p1, p3 - p2)
        dist = area / (np.dot(p1, p2 - p1) + 1e-6)
        return np.abs(dist) < 1e-3

    @staticmethod
    def between(p1, p2, p3):
        # 使用xy平面投影
        p1_2d = p1[:2]
        p2_2d = p2[:2]
        p3_2d = p3[:2]
        if p1_2d[0] != p2_2d[0]:
            return (p1_2d[0] <= p3_2d[0] <= p2_2d[0]) or (p1_2d[0] >= p3_2d[0] >= p2_2d[0])
        else:
            return (p1_2d[1] <= p3_2d[1] <= p2_2d[1]) or (p1_2d[1] >= p3_2d[1] >= p2_2d[1])

    @staticmethod
    def intersect(a, b, c, d):
        # 使用xy平面投影
        a_2d = a[:2]
        b_2d = b[:2]
        c_2d = c[:2]
        d_2d = d[:2]
        if BasicOptions.collinear(a_2d, b_2d, c_2d):
            return BasicOptions.between(a_2d, b_2d, c_2d)
        if BasicOptions.collinear(a_2d, b_2d, d_2d):
            return BasicOptions.between(a_2d, b_2d, d_2d)
        if BasicOptions.collinear(c_2d, d_2d, a_2d):
            return BasicOptions.between(c_2d, d_2d, a_2d)
        if BasicOptions.collinear(c_2d, d_2d, b_2d):
            return BasicOptions.between(c_2d, d_2d, b_2d)
        cd_cross = np.logical_xor(BasicOptions.left_on(a_2d, b_2d, c_2d), BasicOptions.left_on(a_2d, b_2d, d_2d))
        ab_cross = np.logical_xor(BasicOptions.left_on(c_2d, d_2d, a_2d), BasicOptions.left_on(c_2d, d_2d, b_2d))
        return ab_cross and cd_cross
        
    @staticmethod
    def diagonal(verts: np.ndarray, indices: np.ndarray, ia: int, ib: int) -> bool:
        def in_cone(verts: np.ndarray, indices: np.ndarray, ia: int, ib: int) -> bool:
            # Check whether (ia, ib) is in cone of (ia-, ia, ia+)
            n = len(indices)
            ia_prev = ia - 1 if ia - 1 >= 0 else n - 1
            ia_next = ia + 1 if ia + 1 < n else 0

            # turn index of `indices` to index of `verts`
            ia, ib = indices[ia], indices[ib]
            ia_prev, ia_next = indices[ia_prev], indices[ia_next]

            # Convex
            if BasicOptions.left_on(verts[ia_prev], verts[ia], verts[ia_next]):
                return BasicOptions.left_on(verts[ia], verts[ib], verts[ia_prev]) and \
                    BasicOptions.left_on(verts[ib], verts[ia], verts[ia_next])
            # Concave
            return not (BasicOptions.left_on(verts[ia], verts[ib], verts[ia_next]) and \
                        BasicOptions.left_on(verts[ib], verts[ia], verts[ia_prev]))

            
        def diagonalie(verts: np.ndarray, indices: np.ndarray, ia: int, ib: int) -> bool:
            n = len(indices)
            for now_i in range(n):
                # exclude edges contains point a and point b
                if indices[now_i] == indices[ia] or indices[now_i] == indices[ib]:
                    continue
                next_i = (now_i + 1) % n
                if indices[next_i] == indices[ia] or indices[next_i] == indices[ib]:
                    continue

                if BasicOptions.intersect(
                        verts[indices[ia]], verts[indices[ib]],
                        verts[indices[now_i]], verts[indices[next_i]]
                ):
                    return False
            return True
    
        return  in_cone(verts, indices, ia, ib) and \
                in_cone(verts, indices, ib, ia) and \
                diagonalie(verts, indices, ia, ib)

class Geometry_Option:
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
    def merge_holes(verts_poly: np.ndarray, verts_holes: dict[int, np.ndarray]) -> np.ndarray:
        """
        为每个洞找到最短的有效连接线。
        
        Args:
            verts_poly: 外围多边形顶点坐标
            verts_holes: 洞顶点坐标字典，key为洞序号
        
        Returns:
            tuple: (indices_all, diagonals)
                - indices_all: 所有顶点的索引列表
                - diagonals: 连接线列表，每个元素为(hole_vertex_idx, poly_vertex_idx)
        """
        n_poly = len(verts_poly)  # 外围多边形顶点数
        indices_poly = list(range(n_poly))
        indices_holes = {}
        verts_all = verts_poly.copy()
        indices_all = indices_poly.copy()

        # 计算所有洞的顶点索引
        offset = n_poly
        for hole_id, verts_hole in verts_holes.items():
            n_hole = len(verts_hole)
            indices_holes[hole_id] = list(range(offset, offset + n_hole))
            verts_all = np.concatenate((verts_all, verts_hole))
            indices_all.extend(range(offset, offset + n_hole))
            offset += n_hole

        best_diagonals = {}  # 存储每个洞的最佳连接

        # 遍历每个洞
        for hole_id, indices_hole in indices_holes.items():
            verts_hole = verts_holes[hole_id]
            n_hole = len(indices_hole)
            min_diagonal_length = float('inf')
            min_diagonal = None

            # 遍历当前洞的所有顶点
            for hole_idx, hole_vert_idx in enumerate(indices_hole):
                hole_vertex = verts_hole[hole_idx]  # 使用hole_idx而不是hole_vert_idx
                # 检查与外围多边形顶点的连接
                for poly_idx, poly_vertex_idx in enumerate(indices_poly):
                    poly_vertex = verts_poly[poly_idx]
                    okay = True
                    
                    # 检查与外围多边形边的交线
                    for poly_edge in range(n_poly):
                        poly_a = verts_poly[poly_edge]
                        poly_b = verts_poly[(poly_edge + 1) % n_poly]
                        if poly_idx in (poly_edge, (poly_edge + 1) % n_poly):
                            continue
                        if BasicOptions.intersect(poly_vertex, hole_vertex, poly_a, poly_b):
                            okay = False
                            break
                    
                    if not okay:
                        continue
                        
                    # 检查与当前洞边的交线
                    for hole_edge in range(n_hole):
                        hole_a = verts_hole[hole_edge]
                        hole_b = verts_hole[(hole_edge + 1) % n_hole]
                        if hole_idx in (hole_edge, (hole_edge + 1) % n_hole):
                            continue
                        if BasicOptions.intersect(poly_vertex, hole_vertex, hole_a, hole_b):
                            okay = False
                            break
                    
                    if not okay:
                        continue
                        
                    # 检查与其他洞的交线
                    for other_id, other_indices in indices_holes.items():
                        if other_id == hole_id:
                            continue
                        other_verts = verts_holes[other_id]
                        for edge in range(len(other_verts)):
                            a = other_verts[edge]
                            b = other_verts[(edge + 1) % len(other_verts)]
                            if BasicOptions.intersect(poly_vertex, hole_vertex, a, b):
                                okay = False
                                break
                        if not okay:
                            break
                    
                    if okay:
                        # 计算对角线长度
                        diagonal_length = np.linalg.norm(poly_vertex - hole_vertex)
                        if diagonal_length < min_diagonal_length:
                            min_diagonal_length = diagonal_length
                            min_diagonal = (poly_vertex_idx, hole_vert_idx)
            
            if min_diagonal is not None:
                best_diagonals[hole_id] = min_diagonal

        # 构建连接线列表
        diagonals = []
        
        for hole_id, (p_idx, h_idx) in best_diagonals.items():
            diagonals.append((p_idx, h_idx))
        diagonals = sorted(diagonals, key=lambda x: (x[0], -BasicOptions.get_angle_tan(x[0], x[1], verts_all)))

        # 构建新的顶点列表
        verts = []
        for idx in indices_poly:
            verts.append(verts_all[idx])
            
            # 检查是否有从当前顶点出发的连线
            for diagonal in diagonals:
                if diagonal[0] == idx:
                    hole_vertex = diagonal[1]
                    # 找出这个洞顶点属于哪个洞
                    target_hole_id = None
                    target_hole_indices = None
                    for hole_id, hole_indices in indices_holes.items():
                        if hole_vertex in hole_indices:
                            target_hole_id = hole_id
                            target_hole_indices = hole_indices
                            break
                    
                    if target_hole_indices:
                        # 从连线端点开始遍历洞的顶点
                        start_idx = target_hole_indices.index(hole_vertex)
                        n_hole = len(target_hole_indices)
                        # 按顺序添加洞的顶点
                        for i in range(n_hole + 1):  # +1 是为了回到起点
                            current_idx = target_hole_indices[(start_idx + i) % n_hole]
                            verts.append(verts_all[current_idx])
                        # 再次添加当前外围顶点以闭合
                        verts.append(verts_all[idx])
        
        mergelines = [np.array([verts_all[pair[0]], verts_all[pair[1]]]) for pair in diagonals]

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

            list of diagonals that splits the input polygon.
            e.g: [(diag1_a_index, diag1_b_index), ...]
        """

        # find concave vertex
        n = len(indices)
        i_concave = -1

        for ia in range(n):
            ia_prev, ia_next = (ia - 1) % n, (ia + 1) % n
            angle = BasicOptions.angle(verts[indices[ia_prev]], verts[indices[ia]], verts[indices[ia_next]])
            
            if angle > 0:
                pass
            else:
                i_concave = ia
                break

        # if there is no concave vertex, which means current polygon is convex. Return itself directly
        if i_concave == -1:
            return [indices], []

        # Find vertex i_break that `<i_concave, i_break>` is an internal edge
        i_break = -1
        min_diagonal_length = float('inf')  # initialize with infinity
        for i in range(n):
            if i != i_concave and i != (i_concave+1) % n and i != (i_concave-1) % n:
                if BasicOptions.diagonal(verts, indices, i_concave, i):
                    # Calculate the length of the diagonal
                    diagonal_length = np.linalg.norm(verts[indices[i_concave]] - verts[indices[i]])

                    # Update i_break if the current diagonal is shorter
                    if diagonal_length < min_diagonal_length:
                        i_break = i
                        min_diagonal_length = diagonal_length

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
        i1, diag1 = Geometry_Option.split_poly(verts, indices1)
        i2, diag2 = Geometry_Option.split_poly(verts, indices2)

        # merge results from recursively convexify
        ret_diag = [[i_concave, i_break]]
        for diag in diag1:
            ret_diag.append(((diag[0] + i_concave) % n, (diag[1] + i_concave) % n))
        for diag in diag2:
            ret_diag.append(((diag[0] + i_break) % n, (diag[1] + i_break) % n))

        result_indices = i1 + i2

        return result_indices, ret_diag

    @staticmethod
    def split_quad(verts: np.ndarray, indices: np.ndarray) -> Union[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Split a convex polygon into triangles or convex quadrilaterals without obtuse angles.
        
        :param verts: np.ndarray (#verts, 2) - a list of 2D vertices positions
        :param indices: np.ndarray (#verts,) - a list of polygon vertex indices (referring to the array `verts`)
        :return: List of np.ndarray - each sub-array corresponds to the indices of a triangle or quadrilateral
        """
        n = len(indices)
        if n <= 4:
            return [indices]  # Already a triangle or quadrilateral
        
        # Try to form triangles and quadrilaterals
        polygons = []
        
        # Iterate over vertices to create triangles and quadrilaterals
        for i in range(n):
            if i < n - 2:
                # Form a triangle
                triangle_indices = [indices[i], indices[i + 1], indices[i + 2]]
                polygons.append(triangle_indices)
            
            if i < n - 3:
                # Form a quadrilateral if possible
                quad_indices = [indices[i], indices[i + 1], indices[i + 2], indices[i + 3]]
                polygons.append(quad_indices)
        
        return polygons

class MoosasConvexify:
    def plot_faces(faces, lines):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        for face in faces:
            x, y, z = face[:, 0], face[:, 1], face[:, 2]

            x = np.append(x, x[0])
            y = np.append(y, y[0])
            z = np.append(z, z[0])    

            ax.plot(x, y, z, 'purple')  # 绘制多边形的边
            ax.scatter(x, y, z, c='black', marker='o', s=20)

        for line in lines:
            x, y, z = line[:, 0], line[:, 1], line[:, 2] 

            ax.plot(x, y, z, 'blue')  # 绘制多边形的边


        # 设置坐标轴刻度相同
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - 5 * max_range, mid_x + 5 * max_range)
        ax.set_ylim(mid_y - 5 * max_range, mid_y + 5 * max_range)
        ax.set_zlim(mid_z - 2 * max_range, mid_z + 2 * max_range)
        
        plt.show()

    

    def create_quadrilaterals(divide_lines):
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
                
                quad_face = np.array([line1[0], line1[1], line2[1], line2[0]])
                normal_vector = np.cross(line1[0]-line1[1], line1[0]-line2[0])
                quad_normal = normal_vector / np.linalg.norm(normal_vector)

                quad_faces.append(quad_face)
                quad_normals.append(quad_normal)


        
        return quad_faces, quad_normals
    
    def convexify_faces(cat, idd, normal, faces, holes):
        """
        MAIN FUNCTION FOR CONVEXIFY 非凸多边形优化主函数
        1. 读取cat分类、idd序号、normal 法线、faces面节点、holes洞节点
        2. 按照面中节点x+y+z的最小值重排节点起始点，按照法线方向归并所有多边形点序列为逆时针方向
        3. 针对带洞多边形进行重整（未完成）
        4. 多边形凸化，算法This divide-and-conquer methods base on Arkin, Ronald C.'s report (1987).
            "Path planning for a vision-based autonomous robot"
        5. 将凸化的分割线链接为四边形，并赋予新分类为空气墙
        7. 生成新面与旧面的索引关系字典
        6. 输出合并后的cat分类、idd序号、normal 法线、faces面节点
        """
        
        convex_cat = []
        convex_idd = []
        convex_normal = []
        convex_faces = []
        divide_lines = []
        
        for idx, face in enumerate(faces):
            
            is_upward = normal[idx][2] > 0

            if np.abs(normal[idx][2]) > 1e-3:  # wall判断
                poly_ex = Geometry_Option.reorder_vertices(face, is_upward=is_upward)
                
                # 处理多个洞的情况
                if holes[idx]:
                    
                    poly_in = {}
                    for i in range(len(holes[idx])):
                        hole_verts = Geometry_Option.reorder_vertices(holes[idx][i], is_upward=is_upward)
                        poly_in[i] = hole_verts
                    
                    verts, mergelines = Geometry_Option.merge_holes(poly_ex,  poly_in)
                    divide_lines.extend(mergelines)
                    
                else:
                    verts = poly_ex

                # Convexification
                indices = list(range(len(verts)))

                polys, diags = Geometry_Option.split_poly(verts, indices)
                
                subfaces = [verts[poly] for poly in polys]
                
                if len(subfaces) == 1:
                    continue
                else:
                    for i, subface in enumerate(subfaces):
                        convex_cat.append(cat[idx])
                        convex_idd.append(f"#{idd[idx]}")
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

        quad_faces, quad_normals = MoosasConvexify.create_quadrilaterals(divide_lines)
        
        for i,face in enumerate(quad_faces):
            convex_cat.append(2)   #新增分类为空气墙
            convex_idd.append(f"a_{i}")
            convex_normal.append(quad_normals[i])
            convex_faces.append(face)
        

        # Plotting convexified faces
        MoosasConvexify.plot_faces(convex_faces, divide_lines)

        return convex_cat, convex_idd, convex_normal, convex_faces


