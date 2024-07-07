import matplotlib.pyplot as plt
import numpy as np

# 示例数据
categories = ['PACE', 'POT-DRL', 'SPQ', 'QoE-TILE']
qoe = [36.26, 32.15, 28.04, 30.12]
err_qoe = [1.99, 1.81, 1.77, 2.05]  # 人工指定的误差棒高度
quli=[53.93,51.33,48.05,53.17]
quli_err=[1.17,0.97,1,0.98]
rebuf=[1.7,1.844,1.932,2.24]
err_rebuf=[0.111,0.116,0.118,0.136]
s=[0.67,0.74,0.68,0.66]
s_err=[0.02,0.02,0.02,0.02]
# 创建柱状图
fig, axes  =  plt.subplots(2, 2, figsize=(16, 10), sharex='none')

# 画柱状图并添加误差棒
axes[0][0].bar(categories, qoe, yerr=err_qoe, capsize=10, color=['tomato','mediumseagreen','skyblue','sandybrown'], edgecolor='black', width=0.6, hatch = [ "/" , "\\" , "|" , "-" ])
axes[0,0].set_ylabel('QoE score',fontsize = 24)
axes[0,0].xaxis.set_tick_params(labelsize=24)
axes[0,0].yaxis.set_tick_params(labelsize=24)
axes[0,0].set_yticks([20,30,40])
axes[0,0].set_ylim([20, 40])
axes[0,0].set_xlabel('(a)',fontsize = 24)
axes[0,0].grid(True)

axes[0][1].bar(categories, quli, yerr=quli_err, capsize=10, color=['tomato','mediumseagreen','skyblue','sandybrown'], edgecolor='black', width=0.6, hatch = [ "/" , "\\" , "|" , "-" ])
axes[0,1].set_ylabel('Video quality',fontsize = 24)
axes[0,1].xaxis.set_tick_params(labelsize=24)
axes[0,1].yaxis.set_tick_params(labelsize=24)
axes[0,1].set_yticks([40,50,60])
axes[0,1].set_ylim([40, 60])
axes[0,1].set_xlabel('(b)',fontsize = 24)
axes[0,1].grid(True)

axes[1][0].bar(categories, rebuf, yerr=err_rebuf, capsize=10, color=['tomato','mediumseagreen','skyblue','sandybrown'], edgecolor='black', width=0.6, hatch = [ "/" , "\\" , "|" , "-" ])
axes[1,0].set_ylabel('Rebuffering (sec)',fontsize = 24)
axes[1,0].xaxis.set_tick_params(labelsize=24)
axes[1,0].yaxis.set_tick_params(labelsize=24)
axes[1,0].set_yticks([1.6,2.0,2.4])
axes[1,0].set_ylim([1.6,2.4])
axes[1,0].set_xlabel('(c)',fontsize = 24)
axes[1,0].grid(True)

axes[1][1].bar(categories, s, yerr=s_err, capsize=10, color=['tomato','mediumseagreen','skyblue','sandybrown'], edgecolor='black', width=0.6, hatch = [ "/" , "\\" , "|" , "-" ])
axes[1,1].set_ylabel('Quality variation',fontsize = 24)
axes[1,1].xaxis.set_tick_params(labelsize=24)
axes[1,1].yaxis.set_tick_params(labelsize=24)
axes[1,1].set_yticks([0.6,0.7,0.8])
axes[1,1].set_ylim([0.6, 0.8])
axes[1,1].set_xlabel('(d)',fontsize = 24)
axes[1,1].grid(True)
# 显示图形

plt.subplots_adjust(wspace =0.25, hspace =0.35)
plt.savefig('wa.svg', dpi=600, orientation='portrait', format='svg',transparent=False, bbox_inches='tight', pad_inches=0.1)
plt.show()