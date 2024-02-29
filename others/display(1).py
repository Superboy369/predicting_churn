import matplotlib.pyplot as plt
import numpy as np


def plot(kmeans_model=None, columns=None):
    '''
    此函数用户绘制客户分群结果的雷达图
    :param kmeans_model: 聚类的结果
    :param columns: 各特征的明亨
    :return: 客户分群的雷达图
    '''
    plt.figure(figsize=(11, 11))

    # 中文和负号的正常显示
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    plt.style.use('ggplot')  # 使用ggplot的绘图风格
    N = len(columns)         # 特征数
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)  # 设置雷达图的角度，用于平分切开一个圆面
    angles = np.concatenate((angles, [angles[0]]))         # 为了使雷达图一圈封闭起来


    '''
    绘图
    '''

    feature = columns     # 特征名称
    lab = np.concatenate((feature, [feature[0]]))#标签也要封闭

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, polar=True)           # 这里一定要设置为极坐标格式


    ax.set_thetagrids(angles * 180 / np.pi, lab)  # 添加每个特征的标签
    ax.set_ylim(kmeans_model.cluster_centers_.min(), kmeans_model.cluster_centers_.max())  # 设置雷达图的范围
    plt.title('客户群特征分布图')           # 添加标题
    ax.grid(True)                         # 添加网格线
    sam = ['r-', 'o-', 'g-', 'b-', 'p-']  # 样式

    lab = []
    for i in range(len(kmeans_model.cluster_centers_)):  # 依次绘制每个类中心的图像
        values = kmeans_model.cluster_centers_[i]
        values = np.concatenate((values, [values[0]]))   # 为了使雷达图一圈封闭起来，需要下面的步骤
        ax.plot(angles, values, sam[i], linewidth=2)     # 绘制折线图
        ax.fill(angles, values, alpha=0.25)              # 填充颜色
        lab.append('客户群' + str(i))

    plt.legend(lab)
    plt.show()  # 显示图形