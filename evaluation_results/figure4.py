import matplotlib.pyplot as plt
import numpy as np

qoe1 = [35.4,27.63,22.01]
qoe2 = [27.61,24.76,20.76]
qoe3 = [31.15,23.99,17.91]
qoe4 = [29.22,22.87,16.24]
qoe1_errors =[1.71,1.63,1.54]
qoe2_errors =[1.63,1.52,1.31]
qoe3_errors =[2.07,1.88,1.9]
qoe4_errors =[1.72,1.54,1.58]
# 假设有三组数据

quli1 = [53.45,47.05,44.28]
quli2 = [48.79,46.32,41.55]
quli3 = [57.97,50.91,47.4]
quli4 = [53.4,47.16,42.76]
quli1_errors =[1.06,0.89,0.75]
quli2_errors =[0.87,0.81,0.6]
quli3_errors =[0.95,0.81,0.78]
quli4_errors =[0.98,0.8,0.77]

rebuf1 = [1.745,1.878,2.162]
rebuf2 = [2.046,2.079,2.028]
rebuf3 = [2.611,2.62,2.881]
rebuf4 = [2.348,2.355,2.583]
rebuf1_errors =[0.101,0.097,0.103]
rebuf2_errors =[0.107,0.106,0.104]
rebuf3_errors =[0.138,0.137,0.147]
rebuf4_errors =[0.114,0.112,0.126]

smo1 = [0.6,0.64,0.65]
smo2 = [0.71,0.77,0.52]
smo3 = [0.7,0.72,0.68]
smo4 = [0.7,0.74,0.69]
smo1_errors =[0.01,0.02,0.02]
smo2_errors =[0.02,0.02,0.01]
smo3_errors =[0.02,0.02,0.02]
smo4_errors =[0.02,0.02,0.02]
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex='none')

x = ["high", "medium", "low"]
# ax.errorbar(x, y, yerr=errors, fmt='o-', capsize=5, label='Data with error bars')

axes[0,0].errorbar(x, qoe1, label='PACE', color='tomato', yerr=qoe1_errors,fmt='o-', capsize=10, capthick=2,elinewidth=3,zorder=5,markersize=10,linewidth=2)
axes[0,0].errorbar(x, qoe2, label='POT-DRL', linestyle='--', color='mediumseagreen', yerr=qoe2_errors,fmt='*-', capsize=10, capthick=2,elinewidth=3,ms=10,linewidth=2)
axes[0,0].errorbar(x, qoe4, label='SPQ', linestyle=':', color='dodgerblue', yerr=qoe4_errors,fmt='s-', capsize=10, capthick=2,elinewidth=3,ms=10,linewidth=3)
axes[0,0].errorbar(x, qoe3, label='QoE-TILE', linestyle='-.', color='sandybrown', yerr=qoe3_errors,fmt='p-', capsize=10, capthick=2,elinewidth=3,ms=10,linewidth=2)

#axes[0,0].set_xlabel('Bandwidth prediction accuracy',fontsize=20)
axes[0,0].set_ylabel('QoE score',fontsize=24)
axes[0,0].xaxis.set_tick_params(labelsize=24)
axes[0,0].yaxis.set_tick_params(labelsize=24)
axes[0,0].grid(True)
axes[0,0].set_yticks([15,25,35]) 
#axes[0, 0].set_xticks(np.arange(1, 4, 1)) 
# axes[0,0].set_xlim([1, 3])
# axes[0,0].set_ylim([0, 1])

axes[0,1].errorbar(x, quli1, label='PACE', color='tomato', yerr=quli1_errors,fmt='o-', capsize=10, capthick=2,elinewidth=3,zorder=5,ms=10,linewidth=2)
axes[0,1].errorbar(x, quli2, label='POT-DRL', linestyle='--', color='mediumseagreen', yerr=quli2_errors,fmt='*-', capsize=10, capthick=2,elinewidth=3,ms=10,linewidth=2)
axes[0,1].errorbar(x, quli4, label='SPQ', linestyle=':', color='dodgerblue', yerr=quli4_errors,fmt='s-', capsize=10, capthick=2,elinewidth=3,ms=10,linewidth=3)
axes[0,1].errorbar(x, quli3, label='QoE-TILE', linestyle='-.', color='sandybrown', yerr=quli3_errors,fmt='p-', capsize=10, capthick=2,elinewidth=3,ms=10,linewidth=2)
#axes[0,1].set_xlabel('Bandwidth prediction accuracy',fontsize=22)
axes[0,1].set_ylabel('Video quality',fontsize=24)
axes[0,1].xaxis.set_tick_params(labelsize=24)
axes[0,1].yaxis.set_tick_params(labelsize=24)
axes[0,1].grid(True)
axes[0,1].set_yticks([40,50,60]) 
# axes[0,1].set_xlim([0, 120])
# axes[0,1].set_ylim([0, 1])

axes[1,0].errorbar(x, rebuf1, label='PACE', color='tomato', yerr=rebuf1_errors,fmt='o-', capsize=10, capthick=2,elinewidth=3,zorder=5,ms=10,linewidth=2)
axes[1,0].errorbar(x, rebuf2, label='POT-DRL', linestyle='--', color='mediumseagreen', yerr=rebuf2_errors,fmt='*-', capsize=10, capthick=2,elinewidth=3,ms=10,linewidth=2)
axes[1,0].errorbar(x, rebuf4, label='SPQ', linestyle=':', color='dodgerblue', yerr=rebuf4_errors,fmt='s-', capsize=10, capthick=2,elinewidth=3,ms=10,linewidth=3)
axes[1,0].errorbar(x, rebuf3, label='QoE-TILE', linestyle='-.', color='sandybrown', yerr=rebuf3_errors,fmt='p-', capsize=10, capthick=2,elinewidth=3,ms=10,linewidth=2)
axes[1,0].set_xlabel('Bandwidth Prediction Accuracy',fontsize=24,labelpad=14)
axes[1,0].set_ylabel('Rebuffering (sec)',fontsize=24)
axes[1,0].xaxis.set_tick_params(labelsize=24)
axes[1,0].yaxis.set_tick_params(labelsize=24)
axes[1,0].grid(True)
axes[1,0].set_yticks([1.6,2.3,3.0])
#axes[1, 0].set_xticks(np.arange(1, 4, 1)) 
# axes[1,0].set_xlim([0, 10])
# axes[1,0].set_ylim([0, 1])

axes[1,1].errorbar(x, smo1, label='PACE', color='tomato', yerr=smo1_errors,fmt='o-', capsize=10, capthick=2,elinewidth=3,zorder=5,ms=10,linewidth=2)
axes[1,1].errorbar(x, smo2, label='POT-DRL', linestyle='--', color='mediumseagreen', yerr=smo2_errors,fmt='*-', capsize=10, capthick=2,elinewidth=3,ms=10,linewidth=2)
axes[1,1].errorbar(x, smo4, label='SPQ', linestyle=':', color='dodgerblue', yerr=smo4_errors,fmt='s-', capsize=10, capthick=2,elinewidth=3,ms=10,linewidth=3)
axes[1,1].errorbar(x, smo3, label='QoE-TILE', linestyle='-.', color='sandybrown', yerr=smo3_errors,fmt='p-', capsize=10, capthick=2,elinewidth=3,ms=10,linewidth=2)
axes[1,1].set_xlabel('Bandwidth Prediction Accuracy',fontsize=24,labelpad=14)
axes[1,1].set_ylabel('Quality variation',fontsize=24)
axes[1,1].xaxis.set_tick_params(labelsize=24)
axes[1,1].yaxis.set_tick_params(labelsize=24)
axes[1,1].grid(True)
axes[1,1].set_yticks([0.5,0.6,0.7,0.8])
#axes[1,1].set_xticks(np.arange(1, 4, 1)) 
# axes[1,1].set_xlim([0, 1.5])
# axes[1,1].set_ylim([0, 1])
# 添加图例
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.54, 0.98), ncol=4, fontsize=24)

plt.tight_layout(rect=[0, 0, 1, 0.9])  # 调整布局以防止图例与子图重叠
plt.subplots_adjust(wspace =0.29, hspace =0.25)
plt.savefig('wa.svg', dpi=600, orientation='portrait', format='svg',transparent=False, bbox_inches='tight', pad_inches=0.1)
plt.show()