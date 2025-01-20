import os
import pygeos
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET
import convexify
from IO import read_geo, write_geo
from scipy.spatial.transform import Rotation as R



class MoosasGraph:
    """
    图化模块
    用于将建筑空间转为结构化有向图
    1   将空间识别为定向包容盒(oriented bounding box)
        表征参数: 5维向量(length, )

        边表征参数：
            space-face：方向；面属性（floor wall roof）
            face-face：关系（相接、附着、属于                                                                                                                                                                                                                                                                                                                                                                                                                                                                   ）
        需要调整的地方：
            geo_out编号与xml一一对应，所以在重新生成geo后的编号也要调整，需要建立一个geo_out与geo_convex之间的索引字典

            获取新的xml的索引方式：
                感觉得重新调整一下输出xml的格式，先分割面生成新的geo_out再transform，要不然很麻烦
                
                创建面节点（geo_convex）
                创建基于原xml的标准图（）

    """
    def __init__(self):
        """初始化一个空的有向图、空void、空面"""
        self.graph = nx.DiGraph() 
        self.face_graph = nx.Graph()
        self.spaces = []
        self.faces = []
        self.positions = {}
        self.fig = None
        self.ax = None

    def graph_representation_xml(self, geo_path, xml_path):
        """
            解析.xml和关联的.geo文件并构建图
            Args:  
                geo_path(str): *.geo file path
                xml_path(str): *.xml file path
            Returns:
                graph
        """
        faces_id = []
        faces_category =[]
        faces_normal = []
        faces_vertices = []

        spaces_id = []
        spaces_area = []
        spaces_hegiht = []
        spaces_boundary = []
        
        # 0.0   初始化读入文件
        faces_category, faces_id, faces_normal, faces_vertices, faces_holes = read_geo(geo_path)

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 0.1   创建 Uid 和 faceId 映射字典
        dict_u = {}

        for elem in root.findall('face') + root.findall('wall') + root.findall('glazing'):
            
            face_id = elem.find('faceId').text
            uid = elem.find('Uid').text
            
            dict_u[face_id] = uid
    
        # 1.1   Adding face nodes (FROM .geo)
        for i in range(len(faces_id)):
            face = {
                'category': faces_category[i],
                'id': faces_id[i],
                'normal': np.array(faces_normal[i]),
                'vertices': np.array(faces_vertices[i])
            }

            self.faces.append(face)
            face_params = self.create_obb(np.array(faces_vertices[i]), np.array(faces_normal[i]))
            
            self.plot_obb_and_points(face['vertices'], face_params)
            
            
            self.graph.add_node(dict_u[face_id], face_params=face_params)
        
        plt.show()

        # 1.2   Adding face edges
        for face in root.findall('face') + root.findall('wall'):
            
            uid = face.find('Uid').text
            glazingid = face.find('glazingId').text

            if glazingid is not None:
                glazings = glazingid.split()
                for glazing in glazings:
                    # 如果邻接的 glazingid 不为空，则添加边
                    self.graph.add_edge(uid, glazing, attr='glazing')
            neighbors = face.find('neighbor')

            if neighbors is not None:
                for edge in neighbors.findall('edge'):
                    edge_keys = edge.text.split()
                    for key in edge_keys:
                        # 如果邻接的 faceId 不为空，则添加边
                        self.graph.add_edge(uid, key)

        
        # 创建face-space边
        for space in root.findall('space'):
            
            space_id = space.find('id').text.strip() 
            print (space_id)
            space_area = space.find('area')
            space_height = space.find('height')
            space_boundary = space.find('boundary')

            spaces_id.append(space_id)
            spaces_area.append(space_area)
            spaces_hegiht.append(space_height)
            spaces_boundary.append(space_boundary)
            #self.graph.add_node(space_id)  # 将 <id> 添加为图的节点

            # 获取 <boundary> 中的所有 <pt> 坐标点，计算中点
            if space_boundary is not None:
                pts = space_boundary.findall('pt')
                
                coords = []
                for pt in pts:
                    x, y, z = map(float, pt.text.split())  # 将字符串转为浮点数坐标
                    coords.append([x, y, z])
                
                if coords:
                    coords = np.array(coords)
                    center = coords.mean(axis=0)  # 计算中点
                    self.positions[space_id] = center  # 存储空间的中心点坐标
            

            # 查找 <space> 节点下的所有 <neighbor> 节点，添加边
            for neighbor in space.findall('neighbor'):
                neighbor_id = neighbor.text.strip()  # 获取 <neighbor> 中的空间 ID
                face_id = neighbor.attrib.get('Faceid')  # 获取 Faceid 属性
                self.graph.add_edge(space_id, neighbor_id, Faceid=face_id)  # 添加边并附加 Faceid 属性

    def draw_graph_3d(self):
        """绘制图结构的三维表示"""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for face in self.faces:
            
            verts = face['vertices']
            x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

            x = np.append(x, x[0])
            y = np.append(y, y[0])
            z = np.append(z, z[0])    

            ax.plot(x, y, z, 'pink')  # 绘制多边形的边
            ax.scatter(x, y, z, c='grey', marker='o', s=10)
        
        # 从 positions 中获取坐标
        for node, pos in self.positions.items():
            ax.scatter(pos[0], pos[1], pos[2], color='purple', s=50)
            ax.text(pos[0], pos[1], pos[2], node, size=10, zorder=1, color='k')

        # 绘制边
        for edge in self.face_graph.edges():
            start, end = edge
            if start in self.positions and end in self.positions:
                start_pos = self.positions[start]
                end_pos = self.positions[end]
                ax.plot([start_pos[0], end_pos[0]], 
                        [start_pos[1], end_pos[1]], 
                        [start_pos[2], end_pos[2]], color='gray')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def get_nodes(self):
        """获取图中的所有节点"""
        return self.face_graph.nodes(data=True)

    def get_edges(self):
        """获取图中的所有边"""
        return self.face_graph.edges(data=True)
    
    def write_gragh_json (self, json_path):
        
        return

    def plot_points(points):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', label="Points")
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        # 显示图例
        ax.legend()

        # 显示图形
        plt.show()

    def plot_obb_and_points(points, obb_params):
        """
        可视化点云和计算得到的 OBB
        
        参数:
            points: np.ndarray, 点坐标数组，形状为 (N, 3)
            obb_params: dict, 包含 OBB 相关的 15 个参数
                center：OBB的中心点
                scale：OBB的尺寸参数长宽高
                rotation：OBB的相对原点3*3旋转矩阵
        """
        # 提取 OBB 参数
        center = obb_params['center']
        l, w, h = obb_params['scale']
        Rot = obb_params['rotation']

        # 8 个 OBB 角点的偏移量
        offsets = np.array([
            [-l/2, -w/2, -h/2],
            [ l/2, -w/2, -h/2],
            [ l/2,  w/2, -h/2],
            [-l/2,  w/2, -h/2],
            [-l/2, -w/2,  h/2],
            [ l/2, -w/2,  h/2],
            [ l/2,  w/2,  h/2],
            [-l/2,  w/2,  h/2]
        ])

        # 旋转并平移 OBB 角点
        corners = np.dot((np.dot(center, Rot.T) + offsets), Rot)

        # 绘制点云
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', label="Points")

        # 连接 OBB 角点的边来构建立方体
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面边
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面边
            [0, 4], [1, 5], [2, 6], [3, 7]   # 连接顶面和底面
        ]

        # 通过 plot 直接绘制 OBB 的边
        for edge in edges:
            ax.plot([corners[edge[0], 0], corners[edge[1], 0]], 
                    [corners[edge[0], 1], corners[edge[1], 1]], 
                    [corners[edge[0], 2], corners[edge[1], 2]], 
                    color='b')

        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        # 显示图例
        ax.legend()

        # 显示图形
        plt.show()

    def plot_obb_and_points(self, points, obb_params):
        # 提取 OBB 参数
        center = obb_params['center']
        l, w, h = obb_params['scale']
        Rot = obb_params['rotation']

        # 8 个 OBB 角点的偏移量
        offsets = np.array([
            [-l/2, -w/2, -h/2],
            [ l/2, -w/2, -h/2],
            [ l/2,  w/2, -h/2],
            [-l/2,  w/2, -h/2],
            [-l/2, -w/2,  h/2],
            [ l/2, -w/2,  h/2],
            [ l/2,  w/2,  h/2],
            [-l/2,  w/2,  h/2]
        ])

        # 旋转并平移 OBB 角点
        corners = np.dot((np.dot(center, Rot.T) + offsets), Rot)

        # 如果还没有创建图形和坐标轴，则创建它们
        if self.fig is None or self.ax is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')

        # 绘制点云
        self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')

        # 连接 OBB 角点的边来构建立方体
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面边
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面边
            [0, 4], [1, 5], [2, 6], [3, 7]   # 连接顶面和底面
        ]

        # 通过 plot 直接绘制 OBB 的边
        for edge in edges:
            self.ax.plot([corners[edge[0], 0], corners[edge[1], 0]], 
                         [corners[edge[0], 1], corners[edge[1], 1]], 
                         [corners[edge[0], 2], corners[edge[1], 2]], 
                         color='b')

    def show_plot(self):
        # 设置坐标轴标签
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_box_aspect([1, 1, 1])
        # 显示图例
        self.ax.legend()
        # 显示图形
        plt.show()

    def create_obb(self, points, normal, min_scale = 0.1):
        """
        创建点集的定向包围盒 (OBB)，并返回 OBB 参数
        参数:
            points: np.ndarray, (N, 3);
            normal: np.ndarray, (3,);
            min_scale: float，OBB 的最小尺度
        返回:
            obb_params: dict，包含 OBB 相关参数
        """
        # 以z轴和法向量创建相对坐标架
        geometry = pygeos.multipoints(points)
        z_axis = np.array([0,0,1])
        z_r = normal
        
        
        if np.abs(z_r[0]) <= 1e-3 and np.abs(z_r[1]) <= 1e-3:  # 法向量判定                  

            z_r = z_axis
            # 使用 pygeos 计算最小外接矩形（OBB），返回旋转的包围盒
            min_rotated_rectangle = pygeos.minimum_rotated_rectangle(geometry)
            
            # 获取 OBB 参数
            obb_coords = np.array(pygeos.get_coordinates(min_rotated_rectangle, include_z=True)) [:-1] 
            obb_coords = np.nan_to_num(obb_coords, nan=points[0,2])

            x_r = (obb_coords[1] - obb_coords[0])/np.linalg.norm(obb_coords[1] - obb_coords[0])
            y_r = (obb_coords[3] - obb_coords[0])/np.linalg.norm(obb_coords[3] - obb_coords[0])

            rotation = np.array([x_r, y_r, z_r])
            rotation_matrix = R.from_matrix(rotation).as_matrix()

            l = np.linalg.norm(obb_coords[1] - obb_coords[0])
            w = np.linalg.norm(obb_coords[3] - obb_coords[0])
            h = max(np.max(points[:, 2])-np.min(points[:, 2]), min_scale) 

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

            # 反向旋转 OBB 坐标
            original_obb_centroid = np.dot(centroid, rotation_matrix)

        # 返回 OBB 参数
        obb_params = {
            'center': original_obb_centroid,
            'scale': np.array([l,w,h]),
            'rotation': rotation_matrix,
        }

        return obb_params
  
    def graph_representation(self, geo_path):
        
        # 为共享顶点分配唯一索引
        def assign_vertex_indices(faces_vertices):
            vertex_dict = {}
            vertex_index = 0
            faces_with_indices = []
            
            for vertices in faces_vertices:
                indexed_vertices = []
                for vertex in vertices:
                    vertex_tuple = tuple(vertex)
                    if vertex_tuple not in vertex_dict:
                        vertex_dict[vertex_tuple] = vertex_index
                        vertex_index += 1
                    indexed_vertices.append(vertex_dict[vertex_tuple])
                faces_with_indices.append(indexed_vertices)
            return faces_with_indices, vertex_dict

        # 优化后的共享顶点判断函数，使用索引比较
        def shared_vertices_by_index(face1, face2):
            vertices1 = set(face1['vertex_indices'])
            vertices2 = set(face2['vertex_indices'])
            shared_count = len(vertices1 & vertices2)  # 计算交集中的顶点数量
            return shared_count >= 2

    
        faces_category, faces_id, faces_normal, faces_vertices = convexify.read_geo(geo_path)

        
        faces_with_indices, vertex_dict = assign_vertex_indices(faces_vertices)

        # 初始化面集合
        for i in range(len(faces_id)):
            face = {
                'category': faces_category[i],
                'id': faces_id[i],
                'normal': np.array(faces_normal[i]),
                'vertices': np.array(faces_vertices[i]),  # 原始顶点
                'vertex_indices': faces_with_indices[i]    # 顶点索引
            }
            self.faces.append(face)

        # 为每个非 category=1 的面添加节点
        for face in self.faces:
            if face['category'] != 1:
                centroid = np.mean(face['vertices'], axis=0)
                self.face_graph.add_node(face['id'], pos=centroid)
                self.positions[face['id']] = centroid

        # 添加共享顶点的边
        for i, face1 in enumerate(self.faces):
            for j, face2 in enumerate(self.faces):
                if i != j and shared_vertices_by_index(face1, face2):
                    self.face_graph.add_edge(face1['id'], face2['id'])



#main
user_profile = os.environ['USERPROFILE']

input_geo_path = rf"{user_profile}/AppData/Roaming/SketchUp/SketchUp 2022/SketchUp/Plugins/pkpm_moosas/data/geometry/selection0_out.geo"   

input_xml_path = rf"{user_profile}/AppData/Roaming/SketchUp/SketchUp 2022/SketchUp/Plugins/pkpm_moosas/data/geometry/selection0.xml" 

output_geo_path = "results/selection0_convex.geo"


def convex_temp():
    
    cat, idd, normal, faces, holes = read_geo(input_geo_path)
    convex_cat, convex_idd, convex_normal, convex_faces = convexify.MoosasConvexify.convexify_faces(cat, idd, normal, faces, holes)
    write_geo (output_geo_path, convex_cat, convex_idd, convex_normal, convex_faces)
    


def graph_temp():
    graph = MoosasGraph()
    graph.graph_representation_xml(output_geo_path, input_xml_path)  
    graph.draw_graph_3d()


convex_temp()
#graph_temp()