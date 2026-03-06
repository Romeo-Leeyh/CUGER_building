# 几何简化与凸分解流程改进文档

## 概述

本文档描述了对 BuildingConvex 项目的重要改进，主要包括：

1. **创建独立的几何工具模块** (`geometry.py`)
2. **实现三级LOD几何简化功能** (`simplify.py`)  
3. **集成简化流程到主处理管道** (`test.py` 和 `process.py`)

---

## 核心改动

### 1. 新建 `geometry.py` - 几何工具中心库

**位置**: `cuger/__transform/geometry.py`

**包含内容**:
- `GeometryBasic`: 基础几何计算类
  - `angle()`: 计算三点形成的有向角度
  - `get_angle_tan()`: 计算向量的反正切值
  - `polygon_area_2d()`: 使用鞋带公式计算2D多边形面积

- `GeometryValidator`: 几何验证类  
  - `_is_left_on()`: 判断点在线段左侧
  - `_is_collinear()`: 检查三点是否共线
  - `_is_between()`: 检查点是否在两点之间
  - `_is_intersect()`: 检查线段是否相交
  - `_is_valid_face()`: 验证3D多边形面是否有效
  - `_is_same_polygon()`: 判断两个多边形是否相同
  - `_is_diagonal()`: 检查对角线是否有效

- `GeometryOperator`: 几何变换操作类
  - `reorder_vertices()`: 重新排序顶点方向
  - `compute_max_inscribed_quadrilateral()`: 计算最大内接四边形
  - `process_hole()`: 处理孔洞
  - `merge_holes()`: 将孔洞合并到多边形
  - `split_poly()`: 将非凸多边形分解为凸多边形
  - `create_airwalls()`: 从分割线创建气墙

- **OBB工具函数**:
  - `create_obb()`: 创建定向包容盒（Oriented Bounding Box）
  - `obb_to_face_vertices()`: 将OBB参数转换为6个面的顶点

**优势**:
- 集中管理所有几何操作
- 避免代码重复
- 便于维护和扩展
- 清晰的类单一职责原则

---

### 2. 实现 `simplify.py` - 三级LOD几何简化

**位置**: `cuger/__transform/simplify.py`

**核心函数**: `simplify_faces(cat, idd, normal, faces, holes, lod="medium")`

**参数**:
- `cat`: 面的类别ID列表
- `idd`: 面的标识符列表
- `normal`: 面的法向量列表
- `faces`: 面的顶点列表
- `holes`: 面的孔洞列表
- `lod`: 简化级别 - "precise"、"medium" 或 "low"

**三个LOD等级**:

#### **precise (精确)** 
```python
lod="precise"
```
- **处理方式**: 直接返回原始几何数据，无任何简化
- **输出**: 完全相同的输入几何
- **用途**: 需要保留所有细节时使用

#### **medium (中等)** 
```python
lod="medium"
```
- **处理方式**: 按照法向量方向分层提取OBB
  - 第1层: 屋顶/地板（法向量Z分量 > 0.7）
  - 第2层: 墙壁（法向量接近水平）
  - 每层单独生成OBB，返回12个面（2个OBB × 6个面）

- **特点**: 保留基本建筑形态，明显降低复杂度
- **用途**: 平衡细节与效率的场景

#### **low (低)** 
```python
lod="low"
```
- **处理方式**: 所有顶点合并生成单个OBB
  - 计算所有面的平均法向量
  - 基于平均法向量创建统一OBB
  - 返回6个面（单个OBB）

- **特点**: 最大程度简化，仅保留建筑外包
- **用途**: 快速概览或低精度建模

**内部函数**:
- `_simplify_to_single_obb()`: 生成单个OBB
- `_simplify_to_multi_layer_obb()`: 生成多层OBB

---

### 3. 更新 `test.py` - 集成简化流程

**主要改动**:

```python
def process_file(input_geo_path, modelname, lod="medium"):
    # Step 1: 简化几何
    ps.simplify_process(input_geo_path, simplified_geo_path, 
                       figure_path=None, lod=lod)
    
    # Step 2: 凸分解
    ps.convex_process(simplified_geo_path, convex_geo_path, 
                     figure_path=figure_path)
    
    # Step 3/4: 后续处理（可选）
```

**处理流程**:
```
原始GEO文件
    ↓
[简化] 根据LOD级别简化
    ↓
简化后GEO文件 (simplified_{lod}.geo)
    ↓
[凸分解] 转换为凸多边形
    ↓
最终凸化GEO文件 (convex_*.geo)
    ↓
[可选] 图形转换 & 格式导出
```

**多LOD测试**:
```python
lod_levels = ["precise", "medium", "low"]

for lod in lod_levels:
    # 对每个LOD级别处理所有输入文件
    process_file(input_geo_path, basename, lod=lod)
```

---

### 4. 更新 `process.py` - 添加简化导入

**改动**:
```python
from .simplify import simplify_faces
```

确保 `simplify_process()` 能够调用 `simplify_faces()` 函数。

---

### 5. 更新 `convexify.py` - 导入几何类

**改动**:
- 移除所有重复的 `GeometryBasic`、`GeometryValidator`、`GeometryOperator` 类定义
- 添加从 `geometry.py` 的导入

```python
from .geometry import GeometryBasic, GeometryValidator, GeometryOperator
```

---

## 使用指南

### 基本使用

```python
from __transform.simplify import simplify_faces
from graphIO import read_geo, write_geo

# 读取GEO文件
cat, idd, normal, faces, holes = read_geo("input.geo")

# 简化几何（选择LOD级别）
simpl_cat, simpl_idd, simpl_normal, simpl_faces = simplify_faces(
    cat, idd, normal, faces, holes, lod="medium"
)

# 保存简化后的几何
write_geo("output_simplified.geo", simpl_cat, simpl_idd, 
          simpl_normal, simpl_faces)
```

### 完整处理管道

```python
from __transform import process as ps

# Step 1: 简化
ps.simplify_process(
    input_geo_path="building_original.geo",
    output_geo_path="building_simplified.geo",
    figure_path="building_simplified.png",
    lod="medium"
)

# Step 2: 凸化
ps.convex_process(
    input_geo_path="building_simplified.geo",
    output_geo_path="building_convex.geo",
    figure_path="building_convex.png"
)
```

### 运行测试脚本

```bash
cd BuildingConvex/cuger
python verify_modules.py  # 验证所有模块
python test.py            # 运行完整处理流程
```

---

## 技术细节

### OBB（定向包容盒）算法

`create_obb()` 函数实现了OBB生成：

1. **输入**: 点云和法向量
2. **处理**:
   - 检查法向量方向（垂直或水平）
   - 使用pygeos计算最小旋转矩形
   - 建立局部坐标系
   - 计算包容盒中心、尺度、旋转
3. **输出**: OBB参数字典
   ```python
   {
       'center': np.array([x, y, z]),
       'scale': np.array([length, width, height]),
       'rotation': rotation_matrix (3x3)
   }
   ```

### 多层提取算法

`_simplify_to_multi_layer_obb()` 按照建筑结构特征分层：

- **屋顶/地板** (法向量 |z| > 0.7)
- **墙壁** (法向量水平)

每层独立计算OBB，保留建筑的关键特征。

---

## 文件结构

```
BuildingConvex/
├── cuger/
│   ├── __transform/
│   │   ├── geometry.py          [新建] 几何工具库
│   │   ├── simplify.py          [改进] LOD简化模块
│   │   ├── convexify.py         [修改] 导入geometry模块
│   │   ├── process.py           [修改] 添加simplify导入
│   │   └── graph.py             [保持] 图形处理
│   ├── test.py                  [修改] 集成简化流程
│   └── verify_modules.py        [新建] 验证脚本
```

---

## 性能考虑

| LOD级别 | 输出面数 | 计算时间 | 适用场景 |
|---------|---------|---------|---------|
| precise | 原始面数 | 最长 | 高精度分析 |
| medium  | 12 | 中等 | 平衡效率与准确性 |
| low     | 6  | 最短 | 快速概览、可视化 |

---

## 后续扩展

1. **更多LOD级别**: 可添加 "ultra-low"（2面）
2. **智能分层**: 基于面积、曲率等更复杂的分层策略
3. **保留细节特征**: 在简化时保留开窗、阳台等重要特征
4. **性能优化**: 使用空间索引加速OBB计算

---

## 故障排查

### 导入错误
- 确保 `__init__.py` 文件存在于 `__transform` 目录
- 检查Python路径设置

### OBB计算失败
- 检查输入点云是否为空
- 验证法向量是否已归一化

### 几何验证失败
- 检查多边形顶点顺序是否正确
- 确保三维坐标均为有限数值

---

## 许可证

遵循原项目许可证
