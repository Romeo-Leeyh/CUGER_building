import os
import pygeos
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

FACE_PARAM_TEMPLATE = {
    "t": None,
    "v": None,
    "c": None,
    "s": None,
    "r": None,
    "n": None,
    "l": 0,         # 表达是否暴露在外部
}

SPACE_PARAM_TEMPLATE = {
    "c": None,
    "s": None,
    "r": None,
}

def create_obb(points, normal, min_scale = 0.1):
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
        obb_coords[:,2] = (np.min(points[:, 2])+ np.max(points[:, 2]))/2

        # 计算边向量
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
        
            # 计算范数
            x_norm = np.linalg.norm(x_vec)
            y_norm = np.linalg.norm(y_vec)
            
            # 检查范数并计算单位向量
            if x_norm > 1e-6:
                x_r = x_vec / x_norm
            else:
                x_r = np.array([1, 0, 0])  # 默认x方向
                
            if y_norm > 1e-6:
                y_r = y_vec / y_norm
            else:
                y_r = np.array([0, 1, 0])  # 默认y方向

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

def plot_obb_and_points(points, obb_params):
    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
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
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')

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

class MoosasGraph:
    """
    图化模块
    用于将建筑空间转为结构化有向图
    """
    def __init__(self):
        """初始化一个空的有向图、空void、空面"""
        self.graph = nx.Graph() 
        self.spaces = []
        self.faces = []
        self.positions = {}

    def nodes(self):
        """获取图中的所有节点"""
        return self.graph.nodes(data=True)

    def edges(self):
        """获取图中的所有边"""
        return self.graph.edges(data=True)

    def graph_representation_new(self, root, cat, idd, normal, faces, holes):

        dict_u = {}

        # ---------- Face-like nodes ----------
        for face in root.findall('face') + root.findall('wall') + root.findall('glazing') + root.findall('skylight'):
            try:
                uid = face.find('Uid').text
                face_id = face.find('faceId').text
                dict_u[uid] = [face_id]
                self.graph.add_node(
                    uid,
                    node_type="face",
                    face_params=FACE_PARAM_TEMPLATE.copy()
                )
            except Exception as e:
                print(f"[Face Node Error] {e}")
                continue

        # ---------- Space / Void ----------
        for space in root.findall('space'):
            try:
                sid = space.find('id').text.strip()
                is_void = space.find('is_void').text == 'True'
                self.graph.add_node(
                    sid,
                    node_type="void" if is_void else "space",
                    space_params=SPACE_PARAM_TEMPLATE.copy()
                )
            except Exception as e:
                print(f"[Space Node Error] {e}")
                continue

        # ---------- Face-Face edges ----------
        for face in root.findall('face') + root.findall('wall') + root.findall('glazing') + root.findall('skylight'):
            try:
                uid = face.find('Uid').text
                neighbors = face.find('neighbor')
                if neighbors is not None:
                    for edge in neighbors.findall('edge'):
                        for key in edge.text.split():
                            if key in self.graph:
                                self.graph.add_edge(uid, key, adj="adjacent")
            except Exception as e:
                print(f"[Face-Face Edge Error] uid={uid if 'uid' in locals() else None}, {e}")
                continue

        # ---------- Glazing / Shading ----------
        for face in root.findall('face') + root.findall('wall'):
            try:
                uid = face.find('Uid').text
                glazing_element = face.find('glazingId')
                shading_element = face.find('shadingId')

                glazingid = glazing_element.text if glazing_element is not None else None
                shadingid = shading_element.text if shading_element is not None else None

                if glazingid:
                    for g in glazingid.split():
                        if g in self.graph:
                            self.graph.add_edge(uid, g, adj='glazing')

                if shadingid:
                    for s in shadingid.split():
                        if s in self.graph:
                            self.graph.add_edge(uid, s, adj='shading')
            except Exception as e:
                print(f"[Glazing/Shading Error] uid={uid if 'uid' in locals() else None}, {e}")
                continue

        # ---------- Space-Face topology ----------
        for space in root.findall('space'):
            try:
                sid = space.find('id').text.strip()
                topo = space.find('topology')
                if topo is None:
                    print(f"[Skip Space Topology] no topology for space {sid}")
                    continue

                floors = topo.findall('floor/face')
                for floor in floors:
                    floor_id = floor.text
                    if floor_id in self.graph.nodes:
                        self.graph.nodes[floor_id]["face_params"]["t"] = "floor"
                        self.graph.add_edge(sid, floor_id, attr='floor', layer=0)
                    else:
                        print(f"Skipping edge addition: Node floor '{floor_id}' does not exist.")

                ceilings = topo.findall('ceiling/face')
                for ceiling in ceilings:
                    ceiling_id = ceiling.text
                    if ceiling_id in self.graph.nodes:
                        self.graph.nodes[ceiling_id]["face_params"]["t"] = "floor"
                        self.graph.add_edge(sid, ceiling_id, attr='ceiling', layer=0)
                    else:
                        print(f"Skipping edge addition: Node ceiling '{ceiling_id}' does not exist.")

                walls = topo.findall('edge/wall')
                for wall in walls:
                    wall_id = wall.find('Uid').text
                    if wall_id in self.graph.nodes:
                        self.graph.nodes[wall_id]["face_params"]["t"] = "wall"
                        self.graph.add_edge(sid, wall_id, attr='wall', layer=0)
                    else:
                        print(f"Skipping edge addition: Node wall '{wall_id}' does not exist.")

            except Exception as e:
                print(f"[Space-Face Topology Error] sid={sid if 'sid' in locals() else None}, {e}")
                continue

        # ---------- Build face / space parameters ----------
        for nodeid, node in self.graph.nodes(data=True):

            # ---------- Face params ----------
            if node.get("node_type") == "face":
                try:
                    face_id = dict_u.get(nodeid, [None])[0]
                    if face_id not in idd:
                        print(f"[Skip Face] face_id not in idd: {face_id}")
                        continue

                    i = idd.index(face_id)

                    verts = np.array(faces[i])
                    n = np.array(normal[i])

                    obb = create_obb(verts, n)

                    node["face_params"].update({
                        "v": verts,
                        "c": obb["center"],
                        "s": obb["scale"],
                        "r": obb["rotation"],
                        "n": n,
                    })

                    c = int(float(cat[i]))
                    if c == 2:
                        node["face_params"]['t'] = "airwall"
                    elif c in (1, 5, 6):
                        node["face_params"]['t'] = "window"

                except Exception as e:
                    print(f"[Face Param Error] nodeid={nodeid}, {e}")
                    continue

            # ---------- Space / Void params ----------
            if node.get("node_type") in ("space", "void"):
                try:
                    boundary_verts = []

                    for fid in self.graph.neighbors(nodeid):
                        edata = self.graph.get_edge_data(nodeid, fid)
                        if edata is None:
                            continue
                        attr = edata.get("attr")
                        if attr in ("floor", "ceiling", "wall"):
                            face_params = self.graph.nodes[fid].get("face_params", {})
                            if face_params.get("v") is not None:
                                boundary_verts.append(face_params["v"])

                    if not boundary_verts:
                        print(f"[Skip Space] no boundary faces: {nodeid}")
                        continue

                    verts = np.concatenate(boundary_verts, axis=0)
                    obb = create_obb(verts, np.array([0, 0, 1]))

                    node["space_params"].update({
                        "c": obb["center"],
                        "s": obb["scale"],
                        "r": obb["rotation"],
                    })

                except Exception as e:
                    print(f"[Space Param Error] nodeid={nodeid}, {e}")
                    continue

        return self.graph

    def clean_isolated_nodes(self):
        for node in list(self.graph.nodes()):
            if self.graph.degree(node) == 0:
                self.graph.remove_node(node)
                print (f"Removing node: {node}")

    def clean_airwall_nodes(self):
        airwalls = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "face"
            and d.get("face_params", {}).get("t") == "airwall"
        ]

        for airwall in airwalls:
            if airwall not in self.graph:
                continue

            neighbors = list(self.graph.neighbors(airwall))

            for nbr in neighbors:
                if nbr == airwall:
                    continue

                nbr_data = self.graph.nodes[nbr]
                if nbr_data.get("node_type") == "face" and \
                nbr_data.get("face_params", {}).get("t") == "airwall":
                    continue

                # nbr 的所有邻居
                nbr_neighbors = list(self.graph.neighbors(nbr))
                for nn in nbr_neighbors:
                    if nn in (airwall, nbr):
                        continue

                    # 拷贝原边属性
                    edge_data = dict(self.graph.get_edge_data(nbr, nn) or {})

                    # 转接到 airwall
                    if not self.graph.has_edge(airwall, nn):
                        self.graph.add_edge(airwall, nn, **edge_data)

                # 删除 airwall — nbr 边
                if self.graph.has_edge(airwall, nbr):
                    self.graph.remove_edge(airwall, nbr)

                # 删除 nbr 节点（会顺带删掉 nbr—nn）
                if nbr in self.graph:
                    self.graph.remove_node(nbr)
    
    def embed_outer_layer_edges(self, layers: int=3):
        """
        为建筑图结构中的face节点和space-face边分配层级属性
        使用拓扑结构识别外部节点,而不依赖face类型属性
        
        参数:
            layers: int - 需要标记的最大层数
        
        拓扑识别逻辑:
            1. 外部face节点的特征是只与1个space相连(单侧面)
            2. 内部face节点通常与2个space相连(双侧分隔面)
            3. window/airwall节点可能不直接连接space,需要通过adjacent关系传播
            4. 从最外层向内层逐层扩展,直到达到指定深度
        """
        # 初始化所有face节点的层级为0
        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") == "face":
                data["face_params"]["l"] = 0
        
        # 用于记录已处理的节点和空间
        processed_faces = set()
        processed_spaces = set()
        
        # ========== 第1层:找出最外层的face节点 ==========
        # 策略:找出只连接1个space的face节点(拓扑外部特征)
        current_layer_faces = set()
        
        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") == "face":
                # 统计该face节点连接的space/void节点数量
                connected_spaces = []
                for neighbor in self.graph.neighbors(node):
                    neighbor_data = self.graph.nodes[neighbor]
                    if neighbor_data.get("node_type") in ["space", "void"]:
                        connected_spaces.append(neighbor)
                
                # 只连接1个space的face节点被认为是外部节点
                if len(connected_spaces) == 1:
                    current_layer_faces.add(node)
        
        # 扩展:找出与外部节点adjacent的window/airwall节点
        # 这些透明节点虽然可能不直接连接space,但它们在外部
        transparent_nodes = set()
        for face_node in current_layer_faces:
            for neighbor in self.graph.neighbors(face_node):
                neighbor_data = self.graph.nodes[neighbor]
                if neighbor_data.get("node_type") == "face":
                    # 检查是否为透明节点
                    face_type = neighbor_data.get("face_params", {}).get("t")
                    if face_type in ["window", "airwall"]:
                        edge_data = self.graph.get_edge_data(face_node, neighbor)
                        if edge_data and edge_data.get("adj") == "adjacent":
                            transparent_nodes.add(neighbor)
        
        # 将透明节点也加入最外层
        current_layer_faces.update(transparent_nodes)
        
        # 如果没有找到任何外部节点(异常情况),使用备用策略
        if not current_layer_faces:
            # 备用策略:找出度数最小的face节点
            face_degrees = []
            for node, data in self.graph.nodes(data=True):
                if data.get("node_type") == "face":
                    degree = self.graph.degree(node)
                    face_degrees.append((node, degree))
            
            if face_degrees:
                min_degree = min(deg for _, deg in face_degrees)
                current_layer_faces = {node for node, deg in face_degrees if deg == min_degree}
        
        # ========== 开始分层处理 ==========
        for layer in range(1, layers + 1):
            print (f"   Processing layer {layer} with {len(current_layer_faces)} face nodes")
            if not current_layer_faces:
                # 没有更多的face节点可以处理,提前结束
                print ("    No more face nodes to process at layer", layer)
                break
            
            # 为当前层的face节点赋值
            for face_node in current_layer_faces:
                if face_node not in processed_faces:
                    self.graph.nodes[face_node]["face_params"]["l"] = layer
                    processed_faces.add(face_node)
            
            # 找出当前层face节点连接的space节点,并为space-face边添加layer属性
            current_layer_spaces = set()
            for face_node in current_layer_faces:
                for neighbor in self.graph.neighbors(face_node):
                    neighbor_data = self.graph.nodes[neighbor]
                    if neighbor_data.get("node_type") in ["space", "void"]:
                        if neighbor not in processed_spaces:
                            current_layer_spaces.add(neighbor)
                            # 为space-face边添加layer属性
                            if self.graph.has_edge(neighbor, face_node):
                                self.graph[neighbor][face_node]["layer"] = layer
            
            processed_spaces.update(current_layer_spaces)
            
            # ========== 找出下一层的face节点 ==========
            next_layer_faces = set()
            for space_node in current_layer_spaces:
                for neighbor in self.graph.neighbors(space_node):
                    neighbor_data = self.graph.nodes[neighbor]
                    if neighbor_data.get("node_type") == "face":
                        if neighbor not in processed_faces:
                            # 直接添加该face节点
                            next_layer_faces.add(neighbor)
                            
                            # 如果该节点是透明节点,还需要找出与其adjacent的face节点
                            face_type = neighbor_data.get("face_params", {}).get("t")
                            if face_type in ["window", "airwall"]:
                                for trans_neighbor in self.graph.neighbors(neighbor):
                                    trans_neighbor_data = self.graph.nodes[trans_neighbor]
                                    if trans_neighbor_data.get("node_type") == "face":
                                        if trans_neighbor not in processed_faces:
                                            edge_data = self.graph.get_edge_data(neighbor, trans_neighbor)
                                            if edge_data and edge_data.get("adj") == "adjacent":
                                                next_layer_faces.add(trans_neighbor)
            
            # 移动到下一层
            current_layer_faces = next_layer_faces
        
        return self.graph
    
    def graph_edit(self, _isolated_clean=True, _airwall_clean=True, _outer_layer_edge_embedding=True):
        """图结构编辑"""
        if _isolated_clean:
            print("--cleaned isolated nodes--")
            self.clean_isolated_nodes()
 
        if _airwall_clean:
            print("--merged airwall and its parents nodes--")
            self.clean_airwall_nodes()

        if _outer_layer_edge_embedding:
            print("--embedded outer layer edges--")
            self.embed_outer_layer_edges()

        return self.graph

    def draw_graph_3d(self, file_path, _fig_show =False):
        """绘制图结构的三维表示"""
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=45, azim=15)  


        colors = {
            'window': '#FFBE7A',
            'shading': '#999999',
            'floor': '#82B0D2',
            'wall': '#8ECFC9',
            'airwall': '#E7DAD2',
            'space': '#FA7F6F',
            'void': 'white',
            None: 'white'  
        }
        
        # 绘制节点
        for node in self.graph.nodes():
            if 'face_params' in self.graph.nodes[node]:
                center = self.graph.nodes[node]['face_params']['c']
                
                node_type = self.graph.nodes[node]['face_params']['t']

                color = colors.get(node_type, 'brown')
                
                ax.scatter(center[0], center[1], center[2], 
                        c=color, s=25, edgecolors='k')

            if 'space_params' in self.graph.nodes[node]:
                center = self.graph.nodes[node]['space_params']['c']
                if self.graph.nodes[node]['node_type'] == 'void':
                    color = colors['void']
                else:
                    color = colors['space']
                ax.scatter(center[0], center[1], center[2], 
                        c=color, s=50, edgecolors='k')

        
        # 绘制边
        for edge in self.graph.edges():
            start_node, end_node = edge
            if ('face_params' in self.graph.nodes[start_node] and 
                'face_params' in self.graph.nodes[end_node]):
                
                start_pos = self.graph.nodes[start_node]['face_params']['c']
                end_pos = self.graph.nodes[end_node]['face_params']['c']

                # 获取边的属性
                edge_attr = self.graph.edges[edge].get('attr', 'default')
                edge_color = '#999999' if edge_attr == 'default' else 'orange'
                
                # 绘制边
                ax.plot([start_pos[0], end_pos[0]],
                    [start_pos[1], end_pos[1]],
                    [start_pos[2], end_pos[2]],
                    color=edge_color, linestyle='-', alpha=0.05)

            if ('space_params' in self.graph.nodes[start_node] and 
                'face_params' in self.graph.nodes[end_node]):
                
                start_pos = self.graph.nodes[start_node]['space_params']['c']
                end_pos = self.graph.nodes[end_node]['face_params']['c']

                # 获取边的属性
                edge_attr = self.graph.edges[edge].get('attr', 'default')
                edge_color = 'gray' 
                
                # 绘制边
                ax.plot([start_pos[0], end_pos[0]],
                    [start_pos[1], end_pos[1]],
                    [start_pos[2], end_pos[2]],
                    color=edge_color, linestyle='--', alpha=0.5)
                    
        # 添加图例

        x_vals, y_vals, z_vals = [], [], []
    
        for node in self.graph.nodes():
            if 'face_params' in self.graph.nodes[node]:
                center = self.graph.nodes[node]['face_params']['c']
                x_vals.append(center[0])
                y_vals.append(center[1])
                z_vals.append(center[2])

            if 'space_params' in self.graph.nodes[node]:
                center = self.graph.nodes[node]['space_params']['c']
                x_vals.append(center[0])
                y_vals.append(center[1])
                z_vals.append(center[2])   
        # 计算中心点和最大范围，确保 xyz 轴比例一致
        x_mid = (min(x_vals) + max(x_vals)) / 2
        y_mid = (min(y_vals) + max(y_vals)) / 2
        z_mid = (min(z_vals) + max(z_vals)) / 2
        max_range = max(max(x_vals) - min(x_vals), 
                        max(y_vals) - min(y_vals), 
                        max(z_vals) - min(z_vals)) / 2

        ax.set_xlim(x_mid - max_range, x_mid + max_range)
        ax.set_ylim(y_mid - max_range, y_mid + max_range)
        ax.set_zlim(z_mid - max_range, z_mid + max_range)

        # 确保 xyz 轴比例一致


        # 放大显示图形
        ax.dist = 5  # 减小该值可以放大图形，默认值通常是10
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                    markerfacecolor=v, label=k if k else 'face', 
                    markersize=8)
            for k, v in colors.items()
        ]
        #ax.legend(handles=legend_elements)
        
        plt.axis('off')
        ax.set_axis_off()
        #plt.title('Building Graph 3D Visualization')
        if _fig_show:
            plt.show()
        plt.savefig(file_path, dpi = 300)
        plt.close()


    


