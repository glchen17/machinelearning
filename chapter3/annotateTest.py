# !/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from numpy import *


ax1 = plt.subplot(111)

t = arange(0.0, 5.0, 0.01)
s = cos(2*pi*t)
line = plt.plot(t, s, lw=2)
# 被注释的地方xy(x, y)和插入文本的地方xytext(x, y)
# xycoords和textcoords指定xy和xytext的坐标系。默认为data（使用轴域数据坐标系）
#
# | 参数 | 坐标系 |
# | 'figure points' | 距离图形左下角的点数量 |
# | 'figure pixels' | 距离图形左下角的像素数量 |
# | 'figure fraction' | 0,0 是图形左下角，1,1 是右上角 |
# | 'axes points' | 距离轴域左下角的点数量 |
# | 'axes pixels' | 距离轴域左下角的像素数量 |
# | 'axes fraction' | 0,0 是轴域左下角，1,1 是右上角 |
# | 'data' | 使用轴域数据坐标系 |
ax1.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
            arrowprops=dict(facecolor='black'))

plt.ylim(-2, 2)
plt.show()



"""
# Integer subplot specification must be a three digit number
# 前两位代表横竖长度比
# 左边的代表横，中间的代标纵坐标，右边的则表示绘图位置（当横纵比不是1：1时）
ax1, ax2, ax3 = plt.subplot(231), plt.subplot(232), plt.subplot(233)
ax4, ax5, ax6 = plt.subplot(234), plt.subplot(235), plt.subplot(236)

ax2.annotate("Test", xy=(0.5, 0.5), xycoords=ax1.transData,
             xytext=(0.5, 0.5), textcoords=ax2.transData,
             arrowprops=dict(arrowstyle="<-"))
ax3.annotate("ax3", xy=(0.5, 0.5), xytext=(0.5, 0.5))
ax4.annotate("ax4", xy=(0.5, 0.5), xytext=(0.5, 0.5))
ax5.annotate("ax5", xy=(0.5, 0.5), xytext=(0.5, 0.5))
ax6.annotate("ax6", xy=(0.5, 0.5), xytext=(0.5, 0.5))
plt.show()
"""
