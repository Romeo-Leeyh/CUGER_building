import os
import pygeos
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from __transform.convexify import MoosasConvexify
from __transform.graph import MoosasGraph
from graphIO import read_geo, write_geo, read_xml, graph_to_json, json_to_graph
import __transform.process as ps
import sys

def draw_graph_3d(graph, file_path, _fig_show =False):
    """绘制图结构的三维表示"""
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=15)  # 设置仰角为30度，方位角为45度


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
    for node in graph.nodes():
        if 'face_params' in graph.nodes[node]:
            center = graph.nodes[node]['face_params']['c']
            
            node_type = graph.nodes[node]['face_params']['t']

            color = colors.get(node_type, 'brown')
            
            ax.scatter(center[0], center[1], center[2], 
                    c=color, s=25, edgecolors='k')

        if 'space_params' in graph.nodes[node]:
            center = graph.nodes[node]['space_params']['c']
            if graph.nodes[node]['node_type'] == 'void':
                color = colors['void']
            else:
                color = colors['space']
            ax.scatter(center[0], center[1], center[2], 
                    c=color, s=50, edgecolors='k')

    
    # 绘制边
    for edge in graph.edges():
        start_node, end_node = edge
        if ('face_params' in graph.nodes[start_node] and 
            'face_params' in graph.nodes[end_node]):
            
            start_pos = graph.nodes[start_node]['face_params']['c']
            end_pos = graph.nodes[end_node]['face_params']['c']

            # 获取边的属性
            edge_attr = graph.edges[edge].get('attr', 'default')
            edge_color = '#999999' if edge_attr == 'default' else 'orange'
            
            # 绘制边
            ax.plot([start_pos[0], end_pos[0]],
                [start_pos[1], end_pos[1]],
                [start_pos[2], end_pos[2]],
                color=edge_color, linestyle='-', alpha=0.05)

        if ('space_params' in graph.nodes[start_node] and 
            'face_params' in graph.nodes[end_node]):
            
            start_pos = graph.nodes[start_node]['space_params']['c']
            end_pos = graph.nodes[end_node]['face_params']['c']

            # 获取边的属性
            edge_attr = graph.edges[edge].get('attr', 'default')
            edge_color = 'gray' 
            
            # 绘制边
            ax.plot([start_pos[0], end_pos[0]],
                [start_pos[1], end_pos[1]],
                [start_pos[2], end_pos[2]],
                color=edge_color, linestyle='--', alpha=0.5)
                
    # 添加图例

    x_vals, y_vals, z_vals = [], [], []

    for node in graph.nodes():
        if 'face_params' in graph.nodes[node]:
            center = graph.nodes[node]['face_params']['c']
            x_vals.append(center[0])
            y_vals.append(center[1])
            z_vals.append(center[2])

        if 'space_params' in graph.nodes[node]:
            center = graph.nodes[node]['space_params']['c']
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
    plt.savefig(file_path)
    plt.close()


def plot_faces(faces, lines, file_path, _fig_show =False):
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=15)

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

    plt.savefig(file_path)
    if _fig_show:
        plt.show()
    plt.close()


_fig_show = True
input_geo_dir = "BuildingConvex/data/geo/selection0.geo"
output_geo_dir = "BuildingConvex/data/geo/selection0_geo.png"
input_graph_dir = "BuildingConvex/data/graph/selection0"
output_graph_dir = "BuildingConvex/data/selection0_graph.png"

cat, idd, normal, faces, holes = read_geo(input_geo_dir)
plot_faces(faces, file_path=output_geo_dir, _fig_show=_fig_show)

G = json_to_graph(input_graph_dir)
draw_graph_3d(G, file_path=output_graph_dir, _fig_show=_fig_show)

