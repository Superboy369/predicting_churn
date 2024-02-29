import os, re
from PIL import Image
import numpy as np

path = 'D:\\@我的记录文件夹\\生产实习python深度学习\\water_images\\'

def var(rd):    # 求颜色通道的三阶颜色矩
    mid = np.mean((rd-rd.mean())**3)
    return np.sign(mid)*abs(mid)**(1/3)


def get_img_names(path=path):
    file_names = os.listdir(path)
    img_names = []
    for i in file_names:
        if re.findall('^\d_\d+\.jpg$', i) != []:
            img_names.append(i)
    return img_names


def get_img_data(path=path):
    img_names = get_img_names(path=path)
    n = len(img_names)
    data = np.zeros([n, 9])
    labels = np.zeros([n])
    for i in range(n):
        img = Image.open(path+img_names[i])  # 读取图片数据
        M, N = img.size                      # 像素矩阵的行列数
        region = img.crop((M/2-50, N/2-50, M/2+50, N/2+50))  # 截取图像的中心区域

        r, g, b = region.split()   # 分割像素通道
        rd = np.asarray(r)    # 将图片数据转换为数组
        gd = np.asarray(g)
        bd = np.asarray(b)

        data[i, 0] = rd.mean()   # 一阶颜色矩
        data[i, 1] = gd.mean()
        data[i, 2] = bd.mean()

        data[i, 3] = rd.std()    # 二阶颜色矩
        data[i, 4] = gd.std()
        data[i, 5] = bd.std()

        data[i, 6] = var(rd)     # 三阶颜色矩
        data[i, 7] = var(gd)
        data[i, 8] = var(bd)

        labels[i] = img_names[i][0]
    return data, labels
