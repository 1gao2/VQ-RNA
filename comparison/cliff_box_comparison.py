import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.ticker import MultipleLocator
import sys
sys.path.append('../')
plt.rcParams['svg.fonttype'] = 'none'

def acc():
    # 设置全局字体为 Arial
    plt.rcParams['font.family'] = 'Arial'

    # 读取CSV文件
    df = pd.read_csv('cliff_ACC.csv')

    # 设置绘图的尺寸
    plt.figure(figsize=(10, 5))

    # 将数据转换为长格式，以便绘制箱线图
    df_melted = df.melt(id_vars=["Dataset"], var_name="Method", value_name="Accuracy")

    # 为每种方法分配一个数值，用于颜色映射
    method_to_num = {method: i for i, method in enumerate(df_melted['Method'].unique())}
    df_melted['Method_num'] = df_melted['Method'].map(method_to_num)

    # 创建截取的YlGnBu色系的颜色映射
    original_cmap = plt.get_cmap('YlGnBu')
    colors = original_cmap(np.linspace(0.3, 0.7, 256))  # 截取0.3到0.7范围内的颜色
    reversed_colors = colors[::-1]  # 反转颜色顺序
    custom_cmap = LinearSegmentedColormap.from_list('custom_YlGnBu', reversed_colors)

    # 获取每个方法对应的颜色
    method_colors = {method: to_hex(custom_cmap(method_to_num[method] / len(method_to_num))) for method in
                     method_to_num}

    # 输出每个标签和对应的颜色
    for method, color in method_colors.items():
        print(f"Method: {method}, Color: {color}")

    # 在箱线图上叠加散点图，并使用 YlGnBu 色系设置颜色
    scatter = plt.scatter(x=df_melted['Method_num'], y=df_melted['Accuracy'],
                          c=df_melted['Method_num'], cmap=custom_cmap,
                          alpha=0.6, s=50, marker='o', zorder=1)

    # # 添加颜色条
    # cbar = plt.colorbar(scatter)
    # cbar.set_label('Method')

    # 设置 y 轴刻度的间距为 0.25
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.25))

    # 绘制箱线图
    sns.boxplot(x="Method", y="Accuracy", data=df_melted,
                boxprops=dict(facecolor='none', edgecolor='black', linewidth=2),
                medianprops=dict(color="black", linewidth=2),  # 设置中位数线颜色
                whiskerprops=dict(color="black", linewidth=2),  # 设置须的颜色
                capprops=dict(color="black", linewidth=2),  # 设置顶端线的颜色
                # flierprops=dict(marker='o', color='black', alpha=0.5),
                showfliers=False,  # 去掉异常值标记
                width=0.2,
                zorder=2)  # 设置异常值的样式

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)

    # 设置x轴刻度标签为方法名
    plt.xticks(ticks=np.arange(len(method_to_num)), labels=method_to_num.keys(), rotation=45,
               fontsize=14)

    # 增加y轴刻度标签的字号
    plt.yticks(fontsize=14)

    # 调整子图间距，增加x轴标签的显示空间
    plt.subplots_adjust(bottom=0.2)  # 增加底部间距

    # 设置图表标题和标签
    # plt.title("Boxplot of Methods")
    # plt.xlabel("Method", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)

    # 显示图表
    plt.xticks(rotation=45)  # 旋转x轴标签，避免重叠

    plt.savefig(f"cliff_ACC.svg", format='svg')
    plt.show()

def sp():
    # 设置全局字体为 Arial
    plt.rcParams['font.family'] = 'Arial'

    # 读取CSV文件
    df = pd.read_csv('cliff_SP.csv')

    # 设置绘图的尺寸
    plt.figure(figsize=(10, 5))

    # 将数据转换为长格式，以便绘制箱线图
    df_melted = df.melt(id_vars=["Dataset"], var_name="Method", value_name="Specificity")

    # 为每种方法分配一个数值，用于颜色映射
    method_to_num = {method: i for i, method in enumerate(df_melted['Method'].unique())}
    df_melted['Method_num'] = df_melted['Method'].map(method_to_num)

    # 创建截取的YlGnBu色系的颜色映射
    original_cmap = plt.get_cmap('YlGnBu')
    colors = original_cmap(np.linspace(0.3, 0.7, 256))  # 截取0.3到0.7范围内的颜色
    reversed_colors = colors[::-1]  # 反转颜色顺序
    custom_cmap = LinearSegmentedColormap.from_list('custom_YlGnBu', reversed_colors)

    # 获取每个方法对应的颜色
    method_colors = {method: to_hex(custom_cmap(method_to_num[method] / len(method_to_num))) for method in
                     method_to_num}

    # 输出每个标签和对应的颜色
    for method, color in method_colors.items():
        print(f"Method: {method}, Color: {color}")

    # 在箱线图上叠加散点图，并使用 YlGnBu 色系设置颜色
    scatter = plt.scatter(x=df_melted['Method_num'], y=df_melted['Specificity'],
                          c=df_melted['Method_num'], cmap=custom_cmap,
                          alpha=0.6, s=50, marker='o', zorder=1)

    # # 添加颜色条
    # cbar = plt.colorbar(scatter)
    # cbar.set_label('Method')

    # 设置 y 轴刻度的间距为 0.25
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.25))

    # 绘制箱线图
    sns.boxplot(x="Method", y="Specificity", data=df_melted,
                boxprops=dict(facecolor='none', edgecolor='black', linewidth=2),
                medianprops=dict(color="black", linewidth=2),  # 设置中位数线颜色
                whiskerprops=dict(color="black", linewidth=2),  # 设置须的颜色
                capprops=dict(color="black", linewidth=2),  # 设置顶端线的颜色
                # flierprops=dict(marker='o', color='black', alpha=0.5),
                showfliers=False,  # 去掉异常值标记
                width=0.2,
                zorder=2)  # 设置异常值的样式

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)

    # 设置x轴刻度标签为方法名
    plt.xticks(ticks=np.arange(len(method_to_num)), labels=method_to_num.keys(), rotation=45,
               fontsize=14)

    # 增加y轴刻度标签的字号
    plt.yticks(fontsize=14)

    # 调整子图间距，增加x轴标签的显示空间
    plt.subplots_adjust(bottom=0.2)  # 增加底部间距

    # 设置图表标题和标签
    # plt.title("Boxplot of Methods")
    # plt.xlabel("Method", fontsize=14)
    plt.ylabel("Specificity", fontsize=14)

    # 显示图表
    plt.xticks(rotation=45)  # 旋转x轴标签，避免重叠

    plt.savefig(f"cliff_SP.svg", format='svg')
    plt.show()

def se():
    # 设置全局字体为 Arial
    plt.rcParams['font.family'] = 'Arial'

    # 读取CSV文件
    df = pd.read_csv('cliff_SE.csv')

    # 设置绘图的尺寸
    plt.figure(figsize=(10, 5))

    # 将数据转换为长格式，以便绘制箱线图
    df_melted = df.melt(id_vars=["Dataset"], var_name="Method", value_name="Sensitivity")

    # 为每种方法分配一个数值，用于颜色映射
    method_to_num = {method: i for i, method in enumerate(df_melted['Method'].unique())}
    df_melted['Method_num'] = df_melted['Method'].map(method_to_num)

    # 创建截取的YlGnBu色系的颜色映射
    original_cmap = plt.get_cmap('YlGnBu')
    colors = original_cmap(np.linspace(0.3, 0.7, 256))  # 截取0.3到0.7范围内的颜色
    reversed_colors = colors[::-1]  # 反转颜色顺序
    custom_cmap = LinearSegmentedColormap.from_list('custom_YlGnBu', reversed_colors)

    # 获取每个方法对应的颜色
    method_colors = {method: to_hex(custom_cmap(method_to_num[method] / len(method_to_num))) for method in
                     method_to_num}

    # 输出每个标签和对应的颜色
    for method, color in method_colors.items():
        print(f"Method: {method}, Color: {color}")

    # 在箱线图上叠加散点图，并使用 YlGnBu 色系设置颜色
    scatter = plt.scatter(x=df_melted['Method_num'], y=df_melted['Sensitivity'],
                          c=df_melted['Method_num'], cmap=custom_cmap,
                          alpha=0.6, s=50, marker='o', zorder=1)

    # # 添加颜色条
    # cbar = plt.colorbar(scatter)
    # cbar.set_label('Method')

    # 设置 y 轴刻度的间距为 0.25
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.25))

    # 绘制箱线图
    sns.boxplot(x="Method", y="Sensitivity", data=df_melted,
                boxprops=dict(facecolor='none', edgecolor='black', linewidth=2),
                medianprops=dict(color="black", linewidth=2),  # 设置中位数线颜色
                whiskerprops=dict(color="black", linewidth=2),  # 设置须的颜色
                capprops=dict(color="black", linewidth=2),  # 设置顶端线的颜色
                # flierprops=dict(marker='o', color='black', alpha=0.5),
                showfliers=False,  # 去掉异常值标记
                width=0.2,
                zorder=2)  # 设置异常值的样式

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)

    # 设置x轴刻度标签为方法名
    plt.xticks(ticks=np.arange(len(method_to_num)), labels=method_to_num.keys(), rotation=45,
               fontsize=14)

    # 增加y轴刻度标签的字号
    plt.yticks(fontsize=14)

    # 调整子图间距，增加x轴标签的显示空间
    plt.subplots_adjust(bottom=0.2)  # 增加底部间距

    # 设置图表标题和标签
    # plt.title("Boxplot of Methods")
    # plt.xlabel("Method", fontsize=14)
    plt.ylabel("Sensitivity", fontsize=14)

    # 显示图表
    plt.xticks(rotation=45)  # 旋转x轴标签，避免重叠

    plt.savefig(f"cliff_SE.svg", format='svg')
    plt.show()

def mcc():
    # 设置全局字体为 Arial
    plt.rcParams['font.family'] = 'Arial'

    # 读取CSV文件
    df = pd.read_csv('cliff_MCC.csv')

    # 设置绘图的尺寸
    plt.figure(figsize=(10, 5))

    # 将数据转换为长格式，以便绘制箱线图
    df_melted = df.melt(id_vars=["Dataset"], var_name="Method", value_name="MCC")

    # 为每种方法分配一个数值，用于颜色映射
    method_to_num = {method: i for i, method in enumerate(df_melted['Method'].unique())}
    df_melted['Method_num'] = df_melted['Method'].map(method_to_num)

    # 创建截取的YlGnBu色系的颜色映射
    original_cmap = plt.get_cmap('YlGnBu')
    colors = original_cmap(np.linspace(0.3, 0.7, 256))  # 截取0.3到0.7范围内的颜色
    reversed_colors = colors[::-1]  # 反转颜色顺序
    custom_cmap = LinearSegmentedColormap.from_list('custom_YlGnBu', reversed_colors)

    # 在箱线图上叠加散点图，并使用 YlGnBu 色系设置颜色
    scatter = plt.scatter(x=df_melted['Method_num'], y=df_melted['MCC'],
                          c=df_melted['Method_num'], cmap=custom_cmap,
                          alpha=0.6, s=50, marker='o', zorder=1)

    # # 添加颜色条
    # cbar = plt.colorbar(scatter)
    # cbar.set_label('Method')

    # 设置 y 轴刻度的间距为 0.25
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.25))

    # 绘制箱线图
    sns.boxplot(x="Method", y="MCC", data=df_melted,
                boxprops=dict(facecolor='none', edgecolor='black', linewidth=2),
                medianprops=dict(color="black", linewidth=2),  # 设置中位数线颜色
                whiskerprops=dict(color="black", linewidth=2),  # 设置须的颜色
                capprops=dict(color="black", linewidth=2),  # 设置顶端线的颜色
                # flierprops=dict(marker='o', color='black', alpha=0.5),
                showfliers=False,  # 去掉异常值标记
                width=0.2,
                zorder=2)  # 设置异常值的样式

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)

    # 设置x轴刻度标签为方法名
    plt.xticks(ticks=np.arange(len(method_to_num)), labels=method_to_num.keys(), rotation=45,
               fontsize=14)

    # 增加y轴刻度标签的字号
    plt.yticks(fontsize=14)

    # 调整子图间距，增加x轴标签的显示空间
    plt.subplots_adjust(bottom=0.2)  # 增加底部间距

    # 设置图表标题和标签
    # plt.title("Boxplot of Methods")
    # plt.xlabel("Method", fontsize=14)
    plt.ylabel("MCC", fontsize=14)

    # 显示图表
    plt.xticks(rotation=45)  # 旋转x轴标签，避免重叠

    plt.savefig(f"cliff_MCC.svg", format='svg')
    plt.show()

def auc():
    # 设置全局字体为 Arial
    plt.rcParams['font.family'] = 'Arial'

    # 读取CSV文件
    df = pd.read_csv('cliff_AUC.csv')

    # 设置绘图的尺寸
    plt.figure(figsize=(10, 5))

    # 将数据转换为长格式，以便绘制箱线图
    df_melted = df.melt(id_vars=["Dataset"], var_name="Method", value_name="AUC")

    # 为每种方法分配一个数值，用于颜色映射
    method_to_num = {method: i for i, method in enumerate(df_melted['Method'].unique())}
    df_melted['Method_num'] = df_melted['Method'].map(method_to_num)

    # 创建截取的YlGnBu色系的颜色映射
    original_cmap = plt.get_cmap('YlGnBu')
    colors = original_cmap(np.linspace(0.3, 0.7, 256))  # 截取0.3到0.7范围内的颜色
    reversed_colors = colors[::-1]  # 反转颜色顺序
    custom_cmap = LinearSegmentedColormap.from_list('custom_YlGnBu', reversed_colors)

    # 在箱线图上叠加散点图，并使用 YlGnBu 色系设置颜色
    scatter = plt.scatter(x=df_melted['Method_num'], y=df_melted['AUC'],
                          c=df_melted['Method_num'], cmap=custom_cmap,
                          alpha=0.6, s=50, marker='o', zorder=1)

    # # 添加颜色条
    # cbar = plt.colorbar(scatter)
    # cbar.set_label('Method')

    # 设置 y 轴刻度的间距为 0.25
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.25))

    # 绘制箱线图
    sns.boxplot(x="Method", y="AUC", data=df_melted,
                boxprops=dict(facecolor='none', edgecolor='black', linewidth=2),
                medianprops=dict(color="black", linewidth=2),  # 设置中位数线颜色
                whiskerprops=dict(color="black", linewidth=2),  # 设置须的颜色
                capprops=dict(color="black", linewidth=2),  # 设置顶端线的颜色
                # flierprops=dict(marker='o', color='black', alpha=0.5),
                showfliers=False,  # 去掉异常值标记
                width=0.2,
                zorder=2)  # 设置异常值的样式

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)

    # 设置x轴刻度标签为方法名
    plt.xticks(ticks=np.arange(len(method_to_num)), labels=method_to_num.keys(), rotation=45,
               fontsize=14)

    # 增加y轴刻度标签的字号
    plt.yticks(fontsize=14)

    # 调整子图间距，增加x轴标签的显示空间
    plt.subplots_adjust(bottom=0.2)  # 增加底部间距

    # 设置图表标题和标签
    # plt.title("Boxplot of Methods")
    # plt.xlabel("Method", fontsize=14)
    plt.ylabel("AUC", fontsize=14)

    # 显示图表
    plt.xticks(rotation=45)  # 旋转x轴标签，避免重叠

    plt.savefig(f"cliff_AUC.svg", format='svg')
    plt.show()



def cat():
    # 设置全局字体为 Arial
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'Arial'

    # 读取两个CSV文件
    df1 = pd.read_csv('cliff_ACC.csv')
    df2 = pd.read_csv('cliff_SE.csv')
    df3 = pd.read_csv('cliff_SP.csv')
    df4 = pd.read_csv('cliff_AUC.csv')
    df5 = pd.read_csv('cliff_MCC.csv')


    # 为每个文件添加一个标识列，表示数据来自哪个文件
    df1['Source'] = 'File1'
    df2['Source'] = 'File2'
    df3['Source'] = 'File3'
    df4['Source'] = 'File4'
    df5['Source'] = 'File5'

    # 合并两个数据集，并将方法列名改为统一名称
    df1 = df1.melt(id_vars=["Dataset", "Source"], var_name="Method", value_name="Accuracy")
    df2 = df2.melt(id_vars=["Dataset", "Source"], var_name="Method", value_name="Sensitivity")
    df3 = df3.melt(id_vars=["Dataset", "Source"], var_name="Method", value_name="Specificity")
    df4 = df4.melt(id_vars=["Dataset", "Source"], var_name="Method", value_name="AUC")
    df5 = df5.melt(id_vars=["Dataset", "Source"], var_name="Method", value_name="MCC")
    df = pd.concat([df1, df2, df3, df4, df5])


    # 设置绘图的尺寸，将高度缩小
    plt.figure(figsize=(10, 6))  # 宽度为10，高度为4

    # 绘制每个方法的箱线图和散点图，方法对应的图形并列显示
    for i, method in enumerate(df['Method'].unique()):
        df_method = df[df['Method'] == method]

        # 调整 x 轴位置，使两个文件的数据并列显示
        offset = 0.15  # 设置偏移量
        positions = [i - 0.5, i - 0.3, i - 0.1, i + 0.1, i + 0.3]  # 两个箱线图的位置

        # 绘制箱线图，分别为 File1 和 File2 数据
        sns.boxplot(x='Method', y='Accuracy', data=df_method[df_method['Source'] == 'File1'],
                    boxprops=dict(facecolor='none', edgecolor='black', linewidth=1.5),
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(color="black", linewidth=1.5),
                    capprops=dict(color="black", linewidth=1.5),
                    showfliers=False,
                    width=0.2,  # 缩小箱线图宽度
                    positions=[positions[0]],  # 设定位置
                    zorder=2)

        sns.boxplot(x='Method', y='Sensitivity', data=df_method[df_method['Source'] == 'File2'],
                    boxprops=dict(facecolor='none', edgecolor='black', linewidth=1.5),
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(color="black", linewidth=1.5),
                    capprops=dict(color="black", linewidth=1.5),
                    showfliers=False,
                    width=0.2,
                    positions=[positions[1]],
                    zorder=2)

        sns.boxplot(x='Method', y='Specificity', data=df_method[df_method['Source'] == 'File3'],
                    boxprops=dict(facecolor='none', edgecolor='black', linewidth=1.5),
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(color="black", linewidth=1.5),
                    capprops=dict(color="black", linewidth=1.5),
                    showfliers=False,
                    width=0.2,
                    positions=[positions[2]],
                    zorder=2)

        sns.boxplot(x='Method', y='AUC', data=df_method[df_method['Source'] == 'File4'],
                    boxprops=dict(facecolor='none', edgecolor='black', linewidth=1.5),
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(color="black", linewidth=1.5),
                    capprops=dict(color="black", linewidth=1.5),
                    showfliers=False,
                    width=0.2,
                    positions=[positions[3]],
                    zorder=2)

        sns.boxplot(x='Method', y='MCC', data=df_method[df_method['Source'] == 'File5'],
                    boxprops=dict(facecolor='none', edgecolor='black', linewidth=1.5),
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(color="black", linewidth=1.5),
                    capprops=dict(color="black", linewidth=1.5),
                    showfliers=False,
                    width=0.2,
                    positions=[positions[4]],
                    zorder=2)

        # 绘制散点图，分别为 File1 和 File2 数据
        plt.scatter(x=np.full_like(df_method[df_method['Source'] == 'File1']['Accuracy'], positions[0]),
                    y=df_method[df_method['Source'] == 'File1']['Accuracy'],
                    color='#C25759', label='Accuracy' if i == 0 else "",
                    alpha=0.6, s=50, marker='o', zorder=1)

        plt.scatter(x=np.full_like(df_method[df_method['Source'] == 'File2']['Sensitivity'], positions[1]),
                    y=df_method[df_method['Source'] == 'File2']['Sensitivity'],
                    color='#AEB2D1', label='Sensitivity' if i == 0 else "",
                    alpha=0.6, s=50, marker='o', zorder=1)

        plt.scatter(x=np.full_like(df_method[df_method['Source'] == 'File3']['Specificity'], positions[2]),
                    y=df_method[df_method['Source'] == 'File3']['Specificity'],
                    color='#D9B9D4', label='Sensitivity' if i == 0 else "",
                    alpha=0.6, s=50, marker='o', zorder=1)

        plt.scatter(x=np.full_like(df_method[df_method['Source'] == 'File4']['AUC'], positions[3]),
                    y=df_method[df_method['Source'] == 'File4']['AUC'],
                    color='#9FBA95', label='AUC' if i == 0 else "",
                    alpha=0.6, s=50, marker='o', zorder=1)

        plt.scatter(x=np.full_like(df_method[df_method['Source'] == 'File5']['MCC'], positions[4]),
                    y=df_method[df_method['Source'] == 'File5']['MCC'],
                    color='#599CB4', label='MCC' if i == 0 else "",
                    alpha=0.6, s=50, marker='o', zorder=1)


    plt.ylim(0.00, 1.00)
    # plt.yticks(fontsize=24)

    # 设置 y 轴刻度的间距为 0.25
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.25))

    # 去掉图形的上边框和右边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # 设置图形边框的宽度
    # plt.gca().spines['top'].set_linewidth(2)
    # plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)

    # 设置x轴刻度标签为方法名
    plt.xticks(ticks=np.arange(len(df['Method'].unique())), labels=df['Method'].unique(),
               rotation=45,  # 增加旋转角度
               ha='right',  # 右对齐
               fontsize=16)  # 增加x轴刻度标签的字号

    # 增加y轴刻度标签的字号
    plt.yticks(fontsize=16)

    # 调整子图间距，增加x轴标签的显示空间
    plt.subplots_adjust(bottom=0.4)  # 增加底部间距

    # 添加图例，显示散点颜色对应的类别
    plt.legend(loc='upper right', fontsize=16)

    # 设置图表标题和轴标签，并增加字号
    # plt.title("Boxplot with Scatter Plot for Two Files", fontsize=14)
    plt.xlabel("Method", fontsize=16)
    plt.ylabel("Score", fontsize=16)

    # plt.tight_layout()

    plt.savefig(f"cliff_box_comparison.svg", format='svg')

    # 显示图表
    plt.show()


if __name__ == '__main__':
    # rmse()
    # rmse_cliff()
    cat()