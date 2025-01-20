import math
from collections import defaultdict

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def minimum_spanning_tree(polygons):
    # 获取所有顶点
    points = []
    for polygon in polygons:
        points.extend(polygon)
    
    # 构建所有边及其权重
    edges = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            edges.append((distance(points[i], points[j]), i, j))
    
    # 按权重排序
    edges.sort()

    # Kruskal 算法构建最小生成树
    uf = UnionFind(len(points))
    mst = []
    for weight, u, v in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            mst.append((u, v))
    
    return mst, points

def preorder_traversal(tree, start, visited, path):
    visited.add(start)
    path.append(start)
    for neighbor in tree[start]:
        if neighbor not in visited:
            preorder_traversal(tree, neighbor, visited, path)

def tsp_approximation(polygons):
    # 获取最小生成树和顶点
    mst_edges, points = minimum_spanning_tree(polygons)
    
    # 构建邻接表
    tree = defaultdict(list)
    for u, v in mst_edges:
        tree[u].append(v)
        tree[v].append(u)
    
    # 预序遍历获取近似路径
    visited = set()
    path = []
    preorder_traversal(tree, 0, visited, path)  # 从第一个顶点开始遍历
    
    # 返回路径的顶点序列和实际坐标
    path_coordinates = [points[i] for i in path]
    return path, path_coordinates

# 示例多边形
polygons = [
    [(0, 0), (1, 0), (0, 1)],  # 三角形
    [(2, 2), (3, 2), (2, 3)],  # 三角形
    [(4, 0), (5, 0), (4, 1)],  # 三角形
]

# 计算 TSP 近似路径
path_indices, path_coordinates = tsp_approximation(polygons)

# 输出结果
print("路径顶点索引序列:", path_indices)
print("路径坐标序列:", path_coordinates)
