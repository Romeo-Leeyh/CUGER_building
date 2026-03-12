"""
Visualization module for Cuger
Contains all visualization functions for convex faces and graphs
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_convex_faces(faces, lines, file_path, _fig_show=False, overlay_faces=None):
    """
    Plot convexified faces and divide lines in 3D.
    
    Parameters:
    -----------
    faces : list of np.ndarray
        List of convexified face vertices
    lines : list of np.ndarray
        List of divide lines
    file_path : str
        Path to save the figure
    _fig_show : bool
        Whether to show the figure
    overlay_faces : list of np.ndarray, optional
        Optional reference faces to overlay (e.g., original geometry)
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=45, azim=15)
    
    # Remove margins
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if overlay_faces:
        for face in overlay_faces:
            x, y, z = face[:, 0], face[:, 1], face[:, 2]

            x = np.append(x, x[0])
            y = np.append(y, y[0])
            z = np.append(z, z[0])

            ax.plot(x, y, z, color='#777777', alpha=0.35)

    for face in faces:
        x, y, z = face[:, 0], face[:, 1], face[:, 2]

        x = np.append(x, x[0])
        y = np.append(y, y[0])
        z = np.append(z, z[0])    

        ax.plot(x, y, z, 'purple')  
        ax.scatter(x, y, z, c='black', marker='o', s=40)
        
    if lines:
        for line in lines:
            x, y, z = line[:, 0], line[:, 1], line[:, 2] 

            ax.plot(x, y, z, 'blue')  

    all_points = np.vstack(faces)
    if overlay_faces:
        all_points = np.vstack([all_points] + overlay_faces)
    if lines:
        all_points = np.vstack([all_points] + lines) 

    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    z_min, z_max = np.min(all_points[:, 2]), np.max(all_points[:, 2])

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0 * 0.6
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
    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_graph_3d(graph, file_path, _fig_show=False):
    """
    Draw 3D representation of the building graph.
    
    Parameters:
    -----------
    graph : networkx.Graph
        The graph to visualize
    file_path : str
        Path to save the figure
    _fig_show : bool
        Whether to show the figure
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=45, azim=15)
    
    # Remove margins
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

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
    for node in graph.nodes():
        if 'face_params' in graph.nodes[node]:
            center = graph.nodes[node]['face_params']['c']
            
            node_type = graph.nodes[node]['face_params']['t']

            color = colors.get(node_type, 'brown')
            
            ax.scatter(center[0], center[1], center[2], 
                    c=color, s=50, edgecolors='k')

        if 'space_params' in graph.nodes[node]:
            center = graph.nodes[node]['space_params']['c']
            if graph.nodes[node]['node_type'] == 'void':
                color = colors['void']
            else:
                color = colors['space']
            ax.scatter(center[0], center[1], center[2], 
                    c=color, s=100, edgecolors='k')

    
    # Draw edges
    for edge in graph.edges():
        start_node, end_node = edge
        if ('face_params' in graph.nodes[start_node] and 
            'face_params' in graph.nodes[end_node]):
            
            start_pos = graph.nodes[start_node]['face_params']['c']
            end_pos = graph.nodes[end_node]['face_params']['c']

            # Get edge attributes
            edge_attr = graph.edges[edge].get('attr', 'default')
            edge_color = '#999999' if edge_attr == 'default' else 'orange'
            
            # Draw edge
            ax.plot([start_pos[0], end_pos[0]],
                [start_pos[1], end_pos[1]],
                [start_pos[2], end_pos[2]],
                color=edge_color, linestyle='-', alpha=0.05)

        if ('space_params' in graph.nodes[start_node] and 
            'face_params' in graph.nodes[end_node]):
            
            start_pos = graph.nodes[start_node]['space_params']['c']
            end_pos = graph.nodes[end_node]['face_params']['c']

            # Get edge attributes
            edge_attr = graph.edges[edge].get('attr', 'default')
            edge_color = 'gray' 
            
            # Draw edge
            ax.plot([start_pos[0], end_pos[0]],
                [start_pos[1], end_pos[1]],
                [start_pos[2], end_pos[2]],
                color=edge_color, linestyle='--', alpha=0.5)
                
        # Add legend

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
    
    # Compute center and max range to ensure equal axis scales
    x_mid = (min(x_vals) + max(x_vals)) / 2
    y_mid = (min(y_vals) + max(y_vals)) / 2
    z_mid = (min(z_vals) + max(z_vals)) / 2
    max_range = max(max(x_vals) - min(x_vals), 
                    max(y_vals) - min(y_vals), 
                    max(z_vals) - min(z_vals)) / 2 * 0.6

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)

    # Adjust view distance to zoom
    ax.dist = 4  # Decrease this value to zoom in; default is typically 10
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
                markerfacecolor=v, label=k if k else 'face', 
                markersize=8)
        for k, v in colors.items()
    ]
    
    plt.axis('off')
    ax.set_axis_off()
    if _fig_show:
        plt.show()
    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

