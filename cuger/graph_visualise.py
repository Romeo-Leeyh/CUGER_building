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






input_dir = "BuildingConvex\data\graph\selection0"
G = json_to_graph(input_dir)

draw_graph_3d(G, file_path="BuildingConvex/data/figure_graph/selection0_graph.png", _fig_show=True)
