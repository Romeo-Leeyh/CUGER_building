import os
import json
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path

from graphIO import json_to_graph

def analyze_graph_properties(G):
    """计算图的性质"""
    properties = {}
    properties["num_nodes"] = G.number_of_nodes()
    properties["num_edges"] = G.number_of_edges()
    
    # 计算平均度
    degrees = [deg for _, deg in G.degree()]
    properties["avg_degree"] = np.mean(degrees) if degrees else 0
    
    # 计算连通性
    if nx.is_connected(G):
        properties["num_components"] = 1
        properties["largest_component_size"] = properties["num_nodes"]
        properties["diameter"] = nx.diameter(G)  # 仅适用于连通图
    else:
        components = list(nx.connected_components(G))
        properties["num_components"] = len(components)
        properties["largest_component_size"] = max(len(comp) for comp in components)
        properties["diameter"] = None  # 不计算直径
    
    # 计算中心性
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
    properties["avg_degree_centrality"] = np.mean(list(degree_centrality.values())) if degree_centrality else 0
    properties["avg_betweenness_centrality"] = np.mean(list(betweenness_centrality.values())) if betweenness_centrality else 0
    
    # 计算图密度
    properties["density"] = nx.density(G)
    
    # 计算聚类系数
    properties["avg_clustering"] = nx.average_clustering(G)
    
    return properties

def batch_process_graphs(root_dir, output_csv):
    """批量处理多个子文件夹中的图，并保存为CSV"""
    results = []
    
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            try:
                G = json_to_graph(subdir_path)
                graph_properties = analyze_graph_properties(G)
                graph_properties["graph_name"] = subdir  # 以子文件夹名作为图的名称
                results.append(graph_properties)
            except Exception as e:
                print(f"处理 {subdir} 时出错: {e}")

    # 保存为CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

# 示例调用
batch_process_graphs("E:/DATA/Moosasbuildingdatasets/graph", "graph_properties.csv")
