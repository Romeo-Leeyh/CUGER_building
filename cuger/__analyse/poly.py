"""
可视化节点组成的线段和面的程序
支持解析和显示几何数据：
- f: 面的定义
- fn: 面的法向量
- fv: 面的顶点坐标
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from typing import Dict, List, Tuple


class PolyVisualizer:
    """多边形可视化工具"""
    
    def __init__(self):
        self.faces = {}  # 存储面的信息：{face_id: {'normal': [...], 'vertices': [[x,y,z], ...]}}
    
    def parse_input(self, text: str) -> None:
        """
        解析输入文本数据
        格式:
        f,{face_id}_{num_vertices}
        fn,nx,ny,nz  (法向量)
        fv,x,y,z     (顶点坐标)
        """
        lines = text.strip().split('\n')
        current_face_id = None
        current_normal = None
        current_vertices = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            
            if parts[0] == 'f':
                # 保存前一个面的信息
                if current_face_id is not None:
                    self.faces[current_face_id] = {
                        'normal': current_normal,
                        'vertices': current_vertices
                    }
                
                # 开始新的面
                current_face_id = parts[1]
                current_normal = None
                current_vertices = []
            
            elif parts[0] == 'fn':
                # 法向量
                current_normal = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            
            elif parts[0] == 'fv':
                # 顶点坐标
                vertex = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                current_vertices.append(vertex)
        
        # 保存最后一个面
        if current_face_id is not None:
            self.faces[current_face_id] = {
                'normal': current_normal,
                'vertices': current_vertices
            }
    
    def parse_file(self, filepath: str) -> None:
        """从文件中解析数据"""
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        self.parse_input(text)
    
    def visualize(self, show_normals: bool = True, show_vertices: bool = True) -> None:
        """
        可视化所有的面和节点
        
        Args:
            show_normals: 是否显示法向量
            show_vertices: 是否显示顶点
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 用于存储所有顶点以计算坐标范围
        all_vertices = []
        
        # 绘制每一个面
        for face_id, face_data in self.faces.items():
            vertices = face_data['vertices']
            normal = face_data['normal']
            
            if len(vertices) < 3:
                continue
            
            all_vertices.extend(vertices)
            
            # 绘制面
            vertices_array = np.array(vertices)
            
            # 创建三角形面（对于多边形，使用多个三角形）
            if len(vertices) == 3:
                # 三角形
                poly = [[vertices[0], vertices[1], vertices[2]]]
            elif len(vertices) == 4:
                # 四边形
                poly = [[vertices[0], vertices[1], vertices[2], vertices[3]]]
            else:
                # 多边形：从第一个顶点到其他所有顶点分解
                poly = []
                for i in range(1, len(vertices) - 1):
                    poly.append([vertices[0], vertices[i], vertices[i+1]])
            
            # 添加多边形集合（不显示内部边，只显示外轮廓）
            poly_collection = Poly3DCollection(poly, alpha=0.5, edgecolor='none', facecolor='cyan')
            ax.add_collection3d(poly_collection)
            
            # 绘制多边形的边界线
            vertices_array = np.array(vertices)
            # 闭合多边形：连接最后一个顶点到第一个顶点
            boundary = np.vstack([vertices_array, vertices_array[0]])
            ax.plot(boundary[:, 0], boundary[:, 1], boundary[:, 2], 'b-', linewidth=2)
            
            # 显示顶点
            if show_vertices:
                ax.scatter(vertices_array[:, 0], vertices_array[:, 1], vertices_array[:, 2],
                          c='red', s=100, marker='o', label=f'Face {face_id} vertices')
                
                # 标注顶点索引
                for i, v in enumerate(vertices_array):
                    ax.text(v[0], v[1], v[2], f'  {i}', fontsize=8)
            
            # 显示面的中心
            center = np.mean(vertices_array, axis=0)
            ax.scatter(center[0], center[1], center[2], c='blue', s=50, marker='s')
            
            # 显示法向量
            if show_normals and normal is not None:
                normal_length = 20
                normal_normalized = normal / np.linalg.norm(normal) * normal_length
                ax.quiver(center[0], center[1], center[2],
                         normal_normalized[0], normal_normalized[1], normal_normalized[2],
                         color='green', arrow_length_ratio=0.2, linewidth=2)
        
        # 设置坐标轴
        if all_vertices:
            all_vertices = np.array(all_vertices)
            min_coords = all_vertices.min(axis=0)
            max_coords = all_vertices.max(axis=0)
            
            ax.set_xlim([min_coords[0], max_coords[0]])
            ax.set_ylim([min_coords[1], max_coords[1]])
            ax.set_zlim([min_coords[2], max_coords[2]])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Polygon Visualization')
        
        plt.tight_layout()
        plt.show()


def visualize_from_string(text: str, show_normals: bool = True, show_vertices: bool = True) -> None:
    """
    从字符串直接可视化
    
    Args:
        text: 输入的几何数据文本
        show_normals: 是否显示法向量
        show_vertices: 是否显示顶点
    """
    visualizer = PolyVisualizer()
    visualizer.parse_input(text)
    visualizer.visualize(show_normals=show_normals, show_vertices=show_vertices)


def visualize_from_file(filepath: str, show_normals: bool = True, show_vertices: bool = True) -> None:
    """
    从文件可视化
    
    Args:
        filepath: 输入文件路径
        show_normals: 是否显示法向量
        show_vertices: 是否显示顶点
    """
    visualizer = PolyVisualizer()
    visualizer.parse_file(filepath)
    visualizer.visualize(show_normals=show_normals, show_vertices=show_vertices)


# 示例使用
if __name__ == "__main__":
    # 示例数据
    sample_data = """
f,0,0_25
fn,-8.50316331168611e-17,1.2181936654025967e-15,-1.0
fv,85.557,22.086,0.0
fv,76.114,40.699,0.0
fv,64.705,34.91,0.0
fv,52.801,58.374,0.0
fv,61.055,62.561,0.0
fv,67.284,65.721,0.0
fv,79.742,72.042,0.0
fv,92.201,78.362,0.0
fv,85.808,90.963,0.0
fv,83.288,95.931,0.0
fv,76.895,108.532,0.0
fv,68.9,124.291,0.0
fv,60.309,133.865,0.0
fv,66.893,139.772,0.0
fv,60.214,147.215,0.0
fv,53.631,141.308,0.0
fv,26.77,117.208,0.0
fv,20.092,124.651,0.0
fv,33.522,136.701,0.0
fv,46.952,148.752,0.0
fv,63.551,163.645,0.0
fv,78.667,177.208,0.0
fv,95.941,192.706,0.0
fv,102.619,185.263,0.0
fv,75.518,160.947,0.0
fv,70.23,156.202,0.0
fv,87.939,136.464,0.0
fv,91.775,128.903,0.0
fv,79.481,122.666,0.0
fv,84.655,112.469,0.0
fv,91.048,99.867,0.0
fv,103.341,106.104,0.0
fv,109.462,109.209,0.0
fv,140.608,125.011,0.0
fv,137.688,130.767,0.0
fv,144.382,134.163,0.0
fv,154.258,114.695,0.0
fv,149.521,107.442,0.0
fv,143.128,120.043,0.0
fv,124.44,110.563,0.0
fv,111.982,104.242,0.0
fv,104.611,100.503,0.0
fv,107.772,94.274,0.0
fv,115.142,98.013,0.0
fv,118.375,91.641,0.0
fv,130.279,68.178,0.0
fv,142.303,44.476,0.0
fv,133.886,30.925,0.0
fv,129.52,23.898,0.0
fv,105.355,11.638,0.0
fv,82.416,0.0,0.0
fv,74.148,16.297,0.0
fh,0,122.683,49.338,0.0
fh,0,107.079,80.096,0.0
fh,0,106.941,80.026,0.0
fh,0,94.547,73.738,0.0
fh,0,77.966,65.326,0.0
fh,0,87.524,46.487,0.0
fh,0,96.967,27.874,0.0
fh,0,112.3,35.235,0.0
;
    """
    
    print("从示例数据可视化...")
    visualize_from_string(sample_data)
