import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import math
from mpl_toolkits.mplot3d import Axes3D

""" 
在这个程序中，以第一次抽到的次数为变量绘制
概率质量函数和累积分布函数的图像 
"""


## 首先测试下原本的十连保底的情况

def tenthGuaranteeProPMF():
    """十连保底的概率质量函数图像的绘制 """
    p = 0.1
    index = [i for i in range(1, 11)]
    pro = [(1 - p) ** i * p for i in range(9)]
    pro.append((1 - p) ** 9)
    cdf = []
    temp = 0
    for p in pro:
        temp += p
        cdf.append(temp)
    expectation = 0
    for i in range(10):
        expectation = expectation + (i + 1) * pro[i]

    plt.rcParams['font.family'] = ['SimHei']
    plt.subplot(1, 2, 1)
    plt.bar(index, pro, label='TenthGuarantee')
    plt.title('minimum tenth guarantee: E[X]:%.4f' % (expectation))
    plt.subplot(1, 2, 2)
    plt.bar(index, cdf, label='CDF')
    plt.title('CDF of minimum tenth guarantee')
    plt.show()


# 没啥问题
# 接下来把这个模型扩展一下

## 搞一个二位的函数 固定次数从1变到100，步长为1, 概率从0.1到1，步长为0.05

def extendGuaranteeProPMF():
    L0 = range(1, 101, 1)
    p0 = np.linspace(0.1, 1.05,num=20)
    E = np.zeros([len(L0), len(p0)])
    for Li in range(len(L0)):
        for pi in range(len(p0)):
            L = L0[Li]
            p = p0[pi]
            index = [i for i in range(1, L)]
            pro = [(1 - p) ** i * p for i in range(L - 1)]
            pro.append((1 - p) ** (L - 1))
            cdf = []
            temp = 0
            for p in pro:
                temp += p
                cdf.append(temp)
            expectation = 0
            for i in range(L):
                expectation = expectation + (i + 1) * pro[i]
            E[Li, pi] = expectation

    fig = plt.figure()
    X,Y=np.meshgrid(L0, p0)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(X, Y, E.T, cmap=plt.cm.winter)

def extendGuaranteeProPMF():
    L0 = range(1, 101, 1)
    p0 = np.linspace(0.1, 1.05,num=20)
    E = np.zeros([len(L0), len(p0)])
    for Li in range(len(L0)):
        for pi in range(len(p0)):
            L = L0[Li]
            p = p0[pi]
            index = [i for i in range(1, L)]
            pro = [(1 - p) ** i * p for i in range(L - 1)]
            pro.append((1 - p) ** (L - 1))
            cdf = []
            temp = 0
            for p in pro:
                temp += p
                cdf.append(temp)
            expectation = 0
            for i in range(L):
                expectation = expectation + (i + 1) * pro[i]
            E[Li, pi] = expectation

    fig = plt.figure()
    X,Y=np.meshgrid(L0, p0)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(X, Y, E.T, cmap=plt.cm.winter)


def GuaranteeCritProPMF():
    p0 = np.linspace(0.1, 1.05, num=20)
    E = np.zeros([len(p0)])
    for pi in range(len(p0)):
        L = 3
        p = p0[pi]
        index = [i for i in range(1, L)]
        pro = [(1 - p) ** i * p for i in range(L - 1)]
        pro.append((1 - p) ** (L - 1))
        cdf = []
        temp = 0
        for p in pro:
            temp += p
            cdf.append(temp)
        expectation = 0
        for i in range(L):
            expectation = expectation + (i + 1) * pro[i]
        E[pi] = expectation

    plt.plot(p0,E)
def main():
    # plt.figure()
    # tenthGuaranteeProPMF()
    # extendGuaranteeProPMF()
    GuaranteeCritProPMF()
    plt.show()


if __name__ == '__main__':
    main()
