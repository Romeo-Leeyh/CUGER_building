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
    "l": 0,         # indicates whether the face is exposed to the outside
}

SPACE_PARAM_TEMPLATE = {
    "c": None,
    "s": None,
    "r": None,
    "l": 0,         # indicates whether the face is exposed to the outside
}

def create_obb(points, normal, min_scale = 0.1):
    """
    Create an oriented bounding box (OBB) for a set of points and return OBB parameters.
    Parameters:
        points: np.ndarray, (N, 3)
        normal: np.ndarray, (3,)
        min_scale: float, minimum allowed OBB scale
    Returns:
        obb_params: dict containing OBB parameters
    """
    # Create a local coordinate frame using the z-axis and the provided normal
    geometry = pygeos.multipoints(points)
    z_axis = np.array([0,0,1])
    z_r = normal
    
    
    if np.abs(z_r[0]) <= 1e-3 and np.abs(z_r[1]) <= 1e-3:  # check normal near z-axis

        z_r = z_axis
        # Use pygeos to compute the minimum rotated rectangle (2D OBB projection)
        min_rotated_rectangle = pygeos.minimum_rotated_rectangle(geometry)
        
        # Get OBB coordinates
        obb_coords = np.array(pygeos.get_coordinates(min_rotated_rectangle, include_z=True)) [:-1] 
        obb_coords = np.nan_to_num(obb_coords, nan=points[0,2])
        obb_coords[:,2] = (np.min(points[:, 2])+ np.max(points[:, 2]))/2

        # Compute edge vectors
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
        
            # Compute norms
            x_norm = np.linalg.norm(x_vec)
            y_norm = np.linalg.norm(y_vec)
            
            # Check norms and compute unit vectors, fallback to defaults if degenerate
            if x_norm > 1e-6:
                x_r = x_vec / x_norm
            else:
                x_r = np.array([1, 0, 0])  # default x direction
                
            if y_norm > 1e-6:
                y_r = y_vec / y_norm
            else:
                y_r = np.array([0, 1, 0])  # default y direction

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
        
        # Inverse-rotate OBB centroid back to original coordinates
        original_obb_centroid = np.dot(centroid, rotation_matrix)

    # Return OBB parameters
    obb_params = {
        'center': original_obb_centroid,
        'scale': np.array([l,w,h]),
        'rotation': rotation_matrix,
    }

    return obb_params

def plot_obb_and_points(points, obb_params):
    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Extract OBB parameters
    center = obb_params['center']
    l, w, h = obb_params['scale']
    Rot = obb_params['rotation']

    # Offsets for 8 OBB corner points
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

    # Rotate and translate OBB corner points
    corners = np.dot((np.dot(center, Rot.T) + offsets), Rot)

    # Plot point cloud
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')

    # Edges connecting OBB corners to form the box
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face edges
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face edges
        [0, 4], [1, 5], [2, 6], [3, 7]   # connecting top and bottom faces
    ]

    # Draw OBB edges using plot
    for edge in edges:
        ax.plot([corners[edge[0], 0], corners[edge[1], 0]], 
                [corners[edge[0], 1], corners[edge[1], 1]], 
                [corners[edge[0], 2], corners[edge[1], 2]], 
                color='b')

class MoosasGraph:
    """
    Graph module
    Converts building spaces into a structured graph representation
    """
    def __init__(self):
        """Initialize an empty graph and empty lists for spaces and faces"""
        self.graph = nx.Graph() 
        self.spaces = []
        self.faces = []
        self.positions = {}

    def nodes(self):
        """Get all nodes in the graph"""
        return self.graph.nodes(data=True)

    def edges(self):
        """Get all edges in the graph"""
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

                # all neighbors of nbr
                nbr_neighbors = list(self.graph.neighbors(nbr))
                for nn in nbr_neighbors:
                    if nn in (airwall, nbr):
                        continue

                    # copy original edge attributes
                    edge_data = dict(self.graph.get_edge_data(nbr, nn) or {})

                    # reconnect to airwall
                    if not self.graph.has_edge(airwall, nn):
                        self.graph.add_edge(airwall, nn, **edge_data)

                # remove airwall—nbr edge
                if self.graph.has_edge(airwall, nbr):
                    self.graph.remove_edge(airwall, nbr)

                # remove nbr node (this also removes nbr—nn edges)
                if nbr in self.graph:
                    self.graph.remove_node(nbr)
    
    def embed_outer_layer_edges(self, max_layers: int=3):
        """
        递归识别建筑图的多层外壳节点。
        1. 检测最外围直接与外界接触的节点（face只与一个space连接），并把与这些face连接的space节点的l记为当前层，透明节点l也记为当前层。
        2. 在临时子图中移除所有l为当前层的space节点及其边，以及l为当前层的透明face节点及其边。
        3. 递归处理下一层，直到没有新节点或达到最大层数。
        """
        # 初始化所有节点的l为0
        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") == "face":
                data["face_params"]["l"] = 0
            if data.get("node_type") in ["space", "void"]:
                data["space_params"]["l"] = 0

        G = self.graph
        layer = 1
        # 用于递归的临时子图
        temp_graph = G.copy()
        while layer <= max_layers:
            # 1. 找到当前子图中所有只与一个space节点连接的face节点
            current_layer_faces = set()
            for node, data in temp_graph.nodes(data=True):
                if data.get("node_type") == "face":
                    connected_spaces = [nbr for nbr in temp_graph.neighbors(node)
                                        if temp_graph.nodes[nbr].get("node_type") == "space"]
                    if len(connected_spaces) == 1:
                        current_layer_faces.add(node)
            # 查找与这些face通过adjacent边直接连接的透明节点
            transparent_nodes = set()
            for face_node in current_layer_faces:
                for neighbor in temp_graph.neighbors(face_node):
                    neighbor_data = temp_graph.nodes[neighbor]
                    if neighbor_data.get("node_type") == "face":
                        face_type = neighbor_data.get("face_params", {}).get("t")
                        if face_type in ["window", "airwall"]:
                            edge_data = temp_graph.get_edge_data(face_node, neighbor)
                            if edge_data and edge_data.get("adj") in ["adjacent", "glazing"]:
                                transparent_nodes.add(neighbor)
            current_layer_faces.update(transparent_nodes)
            if not current_layer_faces:
                print ("No more outer layer faces found. Stopping recursion.")
                break

            print(f"   Processing layer {layer} with {len(current_layer_faces)} face nodes")
            # 标记face节点l=layer
            for face_node in current_layer_faces:
                G.nodes[face_node]["face_params"]["l"] = layer
            # 标记与这些face连接的space节点l=layer
            current_layer_spaces = set()
            for face_node in current_layer_faces:
                for neighbor in G.neighbors(face_node):
                    neighbor_data = G.nodes[neighbor]
                    if neighbor_data.get("node_type") == "space":
                        G.nodes[neighbor]["space_params"]["l"] = layer
                        current_layer_spaces.add(neighbor)
            # 标记透明节点l=layer
            for face_node in current_layer_faces:
                face_type = G.nodes[face_node].get("face_params", {}).get("t")
                if face_type in ["window", "airwall"]:
                    G.nodes[face_node]["face_params"]["l"] = layer
            # 标记space-face边layer属性
            for face_node in current_layer_faces:
                for neighbor in G.neighbors(face_node):
                    if G.nodes[neighbor].get("node_type") == "space":
                        if G.has_edge(neighbor, face_node):
                            G[neighbor][face_node]["layer"] = layer
            # 2. 在临时子图中移除所有l为当前层的space节点及其边，以及l为当前层的透明face节点及其边
            remove_nodes = set()
            for node in temp_graph.nodes:
                if temp_graph.nodes[node].get("node_type") == "space":
                    if G.nodes[node]["space_params"]["l"] == layer:
                        remove_nodes.add(node)
                if temp_graph.nodes[node].get("node_type") == "face":
                    face_type = temp_graph.nodes[node].get("face_params", {}).get("t")
                    if face_type in ["window", "airwall"] and G.nodes[node]["face_params"]["l"] == layer:
                        remove_nodes.add(node)
            temp_graph.remove_nodes_from(remove_nodes)
            layer += 1
        return self.graph
    
    def graph_edit(self, _isolated_clean=True, _airwall_clean=True, _outer_layer_edge_embedding=True):
        """Graph structure editing"""
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
        """Draw 3D representation of the graph"""
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
        
        # Draw nodes
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

        
        # Draw edges
        for edge in self.graph.edges():
            start_node, end_node = edge
            if ('face_params' in self.graph.nodes[start_node] and 
                'face_params' in self.graph.nodes[end_node]):
                
                start_pos = self.graph.nodes[start_node]['face_params']['c']
                end_pos = self.graph.nodes[end_node]['face_params']['c']

                # Get edge attributes
                edge_attr = self.graph.edges[edge].get('attr', 'default')
                edge_color = '#999999' if edge_attr == 'default' else 'orange'
                
                # Draw edge
                ax.plot([start_pos[0], end_pos[0]],
                    [start_pos[1], end_pos[1]],
                    [start_pos[2], end_pos[2]],
                    color=edge_color, linestyle='-', alpha=0.05)

            if ('space_params' in self.graph.nodes[start_node] and 
                'face_params' in self.graph.nodes[end_node]):
                
                start_pos = self.graph.nodes[start_node]['space_params']['c']
                end_pos = self.graph.nodes[end_node]['face_params']['c']

                # Get edge attributes
                edge_attr = self.graph.edges[edge].get('attr', 'default')
                edge_color = 'gray' 
                
                # Draw edge
                ax.plot([start_pos[0], end_pos[0]],
                    [start_pos[1], end_pos[1]],
                    [start_pos[2], end_pos[2]],
                    color=edge_color, linestyle='--', alpha=0.5)
                    
            # Add legend

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
        # Compute center and max range to ensure equal axis scales
        x_mid = (min(x_vals) + max(x_vals)) / 2
        y_mid = (min(y_vals) + max(y_vals)) / 2
        z_mid = (min(z_vals) + max(z_vals)) / 2
        max_range = max(max(x_vals) - min(x_vals), 
                        max(y_vals) - min(y_vals), 
                        max(z_vals) - min(z_vals)) / 2

        ax.set_xlim(x_mid - max_range, x_mid + max_range)
        ax.set_ylim(y_mid - max_range, y_mid + max_range)
        ax.set_zlim(z_mid - max_range, z_mid + max_range)

        # Ensure equal aspect ratio for axes


        # Adjust view distance to zoom
        ax.dist = 5  # Decrease this value to zoom in; default is typically 10
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


    


