import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

colors = {
    'yellow': '#FFBE7A',
    'grey': '#999999',
    'blue': '#82B0D2',
    'green': '#8ECFC9',
    'pink': '#E7DAD2',
    'purple': '#BEB8DC',
    'red': '#FA7F6F',
    'void': 'white',
    None: 'white'
}

def plot_faces(faces, file_path, _fig_show =False):
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
        

    all_points = np.vstack(faces)  


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



def calculate_euler (faces):
    edges = set()
    vertices = set()

    for face in faces:
        
        for i in range(len(face)):
            edge = (tuple(face[i].tolist()), tuple(face[(i + 1) % len(face)].tolist()))  
            edge = tuple(sorted(edge))  
            edges.add(edge)
        
        for vertex in face:
            vertices.add(tuple(vertex.tolist())) 

    F = len(faces)
    E = len(edges)
    V = len(vertices)
    Eu = V - E + F

    return F, E, V, Eu

def plot_hist_violin(data):
    # 定义区间边界（10^1 到 10^5）
    bins = [0, 10**2.1, 10**2.3, 10**2.5, 10**2.7, 10**2.9, 10**3.1, 10**3.3, 10**3.5, 10**3.7, 10**3.9, 10**5]
    
    bin_labels = ["$10^{2}$", "$10^{2.2}$", "$10^{2.4}$", "$10^{2.6}$", "$10^{2.8}$", "$10^{3}$", 
                  "$10^{3.2}$", "$10^{3.4}$", "$10^{3.6}$", "$10^{3.8}$", "$10^{4}$"]
    
    # 计算每个区间的计数
    hist_counts, _ = np.histogram(data, bins=bins)
    
    # 计算每个区间的数据分布
    binned_data = [data[(data >= bins[i]) & (data < bins[i+1])] for i in range(len(bins)-1)]
    
    # 计算相对偏移量（相对于区间上下限的几何平均）
    geom_means = [10**2, 10**2.2, 10**2.4, 10**2.6, 10**2.8, 10**3, 10**3.2, 10**3.4, 10**3.6, 10**3.8, 10**4.5]
    mean_offsets = []
    mean_offsets = []
    violin_data = []
    positions = []
    
    for i, bin_data in enumerate(binned_data):
        if len(bin_data) > 2 and not np.all(np.isnan(bin_data)):
            rel_dev = 100*(bin_data - geom_means[i]) / geom_means[i]
            violin_data.extend(rel_dev)
            positions.extend([i] * len(rel_dev))
            offset = np.mean(rel_dev)
        else:
            offset = 0
            positions.append(i)
            violin_data.append(offset)
        mean_offsets.append(offset)
    
    # 创建DataFrame用于violinplot
    df_violin = pd.DataFrame({
        'position': positions,
        'deviation': violin_data
    })
    
    # 创建画布
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # 直方图
    ax1.bar(range(len(bins)-1), hist_counts, color=colors['purple'], alpha=0.7, width=0.6, edgecolor='black', label='Histogram')
    ax1.set_ylabel("Count")

    ax1.set_xticks(range(len(bins)-1))
    ax1.set_xticklabels(bin_labels, rotation=0)
    ax1.set_ylim(0, 80)
    
    # 小提琴图
    if len(df_violin) > 0:  # 确保有数据再画图
        violin = sns.violinplot(data=df_violin, x='position', y='deviation', ax=ax2,
            inner='quartile',  # 显示四分位数统计
            width=0.6, 
            color=colors['red'], 
            edgecolor=colors['red'],
            alpha=0.5,
            linewidth=1)   
        for line in violin.lines:
            line.set_color('black')  # 设置四分位数线和中位数线的颜色
            line.set_linewidth(1)    # 设置线宽
        
        # 设置中位数点的样式
        for l in violin.lines[1::3]:
            l.set_linewidth(1)       # 加粗中位数线
            l.set_linestyle('-')
    ax2.set_ylabel("Relative Deviation(%)")
    ax2.set_ylim(-80, 80)
    
    # 连接平均值
    valid_indices = ~np.isnan(mean_offsets)
    ax2.plot(np.arange(len(bins)-1)[valid_indices], 
             np.array(mean_offsets)[valid_indices], 
             marker='o',           # 圆形标记
             mfc='white',         # 标记填充颜色为白色（空心）
             mec=colors['red'],  # 标记边缘颜色
             ms=6,               # 标记大小
             mew=1.5,            # 标记边缘宽度
             color=colors['grey'], 
             lw=1.5, 
             label='Mean Offset')
    
    # 图例
    ax1.legend(loc='upper left', frameon=True)
    
    # 为小提琴图和平均值线添加单独的图例
    handles, labels = ax2.get_legend_handles_labels()
    # 自定义图例中小提琴图的颜色和透明度
    ax2.legend(handles=handles, labels=labels, loc='upper right', frameon=True)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.show()

