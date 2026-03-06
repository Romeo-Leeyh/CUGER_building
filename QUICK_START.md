# 快速开始指南

## 🚀 5分钟快速上手

### 前置条件
- Python 3.7+
- numpy, matplotlib, pygeos, scipy

### 基础使用

#### 1. 简单的LOD简化
```python
from cuger.__transform.simplify import simplify_faces
from cuger.graphIO import read_geo, write_geo

# 读取文件
cat, idd, normal, faces, holes = read_geo("input.geo")

# 简化（选择LOD）
r_cat, r_idd, r_normal, r_faces = simplify_faces(
    cat, idd, normal, faces, holes, lod="medium"
)

# 保存结果
write_geo("output.geo", r_cat, r_idd, r_normal, r_faces)
```

#### 2. 完整处理流程
```python
from cuger.__transform import process as ps

# 简化 + 凸化两步处理
ps.simplify_process("input.geo", "simplified.geo", lod="medium")
ps.convex_process("simplified.geo", "convex.geo")
```

#### 3. 批量处理多LOD
```python
from cuger.__transform import process as ps

for lod in ["precise", "medium", "low"]:
    input_file = f"input.geo"
    output_file = f"output_{lod}.geo"
    ps.simplify_process(input_file, output_file, lod=lod)
```

---

## 📊 LOD对比

| 特性 | precise | medium | low |
|------|---------|--------|-----|
| 面数 | 原始数量 | ≈12 | 6 |
| 精度 | 最高 | 中等 | 低 |
| 速度 | 慢 | 快 | 最快 |
| 用途 | 精确分析 | 平衡 | 快速查看 |

---

## 🔍 LOD说明

### precise (精确)
```python
lod="precise"
```
- 完全保留原始几何
- 无任何简化处理
- 适用于：需要完整细节、精确分析

### medium (中等)
```python
lod="medium"
```
- 分为屋顶/地板和墙壁两层
- 各层生成独立OBB
- 适用于：保留建筑特征，提高处理速度

图示：
```
┌─────────────────┐
│    屋顶OBB6面   │ ← Layer 1
├─────────────────┤
│    墙壁OBB6面   │ ← Layer 2  
└─────────────────┘
共12个面
```

### low (低)
```python
lod="low"
```
- 整个建筑生成单一OBB
- 保留基本外包物体
- 适用于：快速预览、可视化

图示：
```
┌─────────────────┐
│   整个建筑OBB   │
│     6个面       │
└─────────────────┘
```

---

## 📁 文件位置

新增/更改的文件：

```
BuildingConvex/
├── cuger/
│   ├── __transform/
│   │   ├── geometry.py         ← 新增（几何库）
│   │   ├── simplify.py         ← 改进（简化模块）
│   │   ├── convexify.py        ← 修改（导入）
│   │   └── process.py          ← 修改（导入）
│   ├── test.py                 ← 改进（LOD流程）
│   └── verify_modules.py       ← 新增（验证脚本）
│
├── GEOMETRY_SIMPLIFY_README.md  ← 详细文档
└── IMPLEMENTATION_SUMMARY.md    ← 改动总结
```

---

## ✅ 验证安装

```bash
# 进入目录
cd BuildingConvex/cuger

# 运行验证脚本
python verify_modules.py
```

期望输出：
```
════════════════════════════════════════════════════════
Testing Geometry and Simplify Modules
════════════════════════════════════════════════════════

1. Importing geometry module...
   ✓ Successfully imported geometry classes and functions

2. Importing simplify module...
   ✓ Successfully imported simplify_faces function

3. Importing convexify module...
   ✓ Successfully imported convexify_faces function

4. Importing process module...
   ✓ Successfully imported process module

5. Testing basic geometry functionality...
   - Calculated polygon area: 1.0000
   - Face validation: True
   ✓ Basic geometry operations working correctly

6. Testing simplify_faces function...
   Testing LOD='precise'...
     - Returned 2 faces (expected 2)
   Testing LOD='low'...
     - Returned 6 faces (expected 6 for OBB)
   Testing LOD='medium'...
     - Returned 12 faces
   ✓ simplify_faces function working correctly

════════════════════════════════════════════════════════
✓ All verification tests passed!
════════════════════════════════════════════════════════
```

---

## 🐛 常见问题

### Q: 导入错误？
**A**: 确保在 `BuildingConvex/cuger` 目录中运行，或添加到Python路径：
```python
import sys
sys.path.insert(0, '/path/to/BuildingConvex/cuger')
```

### Q: OBB计算失败？
**A**: 检查：
- 输入点云是否为空
- 法向量是否已归一化
- pygeos是否正确安装

### Q: 输出面数与预期不符？
**A**: 检查输入的面数和法向量方向
- precise: 输出 = 输入
- medium: 输出 ≤ 12 (取决于分层效果)
- low: 输出 = 6 (总是6个面)

### Q: 性能太慢？
**A**: 尝试使用 "low" LOD，或检查输入点云大小

---

## 📚 更多资源

- 详细文档: [GEOMETRY_SIMPLIFY_README.md](./GEOMETRY_SIMPLIFY_README.md)
- 实现总结: [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)
- 源代码: `cuger/__transform/`

---

## 🎯 下一步

1. ✅ 查看验证脚本输出确认安装
2. ✅ 用自己的GEO文件测试各LOD
3. ✅ 根据需求选择合适的LOD
4. ✅ 集成到自己的处理管道

---

**祝您使用愉快！** 🎉
