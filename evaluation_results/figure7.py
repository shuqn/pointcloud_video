import matplotlib.pyplot as plt
import numpy as np

# 示例数据
categories = ['PACE', 'POT-DRL', 'QoE-TILE', 'SPQ']
t = [4.282, 3.17, 1.59253, 1.59257]
err = [0.223, 0.1218, 0.00828, 0.00828]  # 人工指定的误差棒高度
# 创建柱状图
fig, axes  =  plt.subplots(1, 1, figsize=(4, 4), sharex='none')

# 画柱状图并添加误差棒
axes.bar(categories, t, yerr=err, capsize=5, color='skyblue', edgecolor='black')
axes.set_ylabel('Time (s)')

plt.show()
