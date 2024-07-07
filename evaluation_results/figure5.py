import matplotlib.pyplot as plt
import numpy as np

qoe1 = [27.16,41.92,67.28,81.55]
qoe2 = [27.95,39.04,52.77,61.3]
qoe3 = [27.72,38.64,58.08,70.39]
qoe4 = [27.0,38.84,47.81,62.49]
qoe1_errors =[1.15,0.8,1.26,1.04]
qoe2_errors =[0.87,0.75,0.86,0.92]
qoe3_errors =[0.9,1.01,1.05,1.09]
qoe4_errors =[1,0.86,1.12,1.31]
# 假设有三组数据

quli1 = [48.23,52.58,73.38,83.98]
quli2 = [46.95,50.53,58.39,63.72]
quli3 = [51.31,57.81,64.08,72.74]
quli4 = [48.12,55.73,53.77,64.82]
quli1_errors =[0.76,0.78,1.11,1.04]
quli2_errors =[0.59,0.74,0.84,0.93]
quli3_errors =[0.63,0.85,0.96,1.1]
quli4_errors =[0.8,0.9,1.06,1.33]

rebuf1 =[2.038,0.981,0.524,0.159]
rebuf2 =[1.817,1.051,0.466,0.156]
rebuf3 =[2.287,1.829,0.518,0.156]
rebuf4 =[2.034,1.595,0.516,0.156]
rebuf1_errors =[0.053,0.026,0.025,0.003]
rebuf2_errors =[0.059,0.036,0.023,0.003]
rebuf3_errors =[0.05,0.063,0.025,0.003]
rebuf4_errors =[0.054,0.051,0.025,0.003]

smo1 = [0.69,0.85,0.85,0.84]
smo2 = [0.82,0.98,0.96,0.86]
smo3 = [0.71,0.88,0.82,0.8]
smo4 =[0.77,0.94,0.8,0.77]
smo1_errors =[0.02,0.02,0.01,0.01]
smo2_errors =[0.01,0.02,0.02,0.02]
smo3_errors =[0.01,0.02,0.02,0.01]
smo4_errors =[0.02,0.02,0.02,0.01]
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex='none')

x = ["net 1", "net 2", "net 3", "net 4"]
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
axes[0,0].set_yticks([30,55,80]) 
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
axes[0,1].set_yticks([45,65,85]) 
# axes[0,1].set_xlim([0, 120])
# axes[0,1].set_ylim([0, 1])

axes[1,0].errorbar(x, rebuf1, label='PACE', color='tomato', yerr=rebuf1_errors,fmt='o-', capsize=10, capthick=2,elinewidth=3,zorder=5,ms=10,linewidth=2)
axes[1,0].errorbar(x, rebuf2, label='POT-DRL', linestyle='--', color='mediumseagreen', yerr=rebuf2_errors,fmt='*-', capsize=10, capthick=2,elinewidth=3,ms=10,linewidth=2)
axes[1,0].errorbar(x, rebuf4, label='SPQ', linestyle=':', color='dodgerblue', yerr=rebuf4_errors,fmt='s-', capsize=10, capthick=2,elinewidth=3,ms=10,linewidth=3)
axes[1,0].errorbar(x, rebuf3, label='QoE-TILE', linestyle='-.', color='sandybrown', yerr=rebuf3_errors,fmt='p-', capsize=10, capthick=2,elinewidth=3,ms=10,linewidth=2)
axes[1,0].set_xlabel('Network Index',fontsize=24,labelpad=14)
axes[1,0].set_ylabel('Rebuffering (sec)',fontsize=24)
axes[1,0].xaxis.set_tick_params(labelsize=24)
axes[1,0].yaxis.set_tick_params(labelsize=24)
axes[1,0].grid(True)
axes[1,0].set_yticks([0,1,2])
#axes[1, 0].set_xticks(np.arange(1, 4, 1)) 
# axes[1,0].set_xlim([0, 10])
# axes[1,0].set_ylim([0, 1])

axes[1,1].errorbar(x, smo1, label='PACE', color='tomato', yerr=smo1_errors,fmt='o-', capsize=10, capthick=2,elinewidth=3,zorder=5,ms=10,linewidth=2)
axes[1,1].errorbar(x, smo2, label='POT-DRL', linestyle='--', color='mediumseagreen', yerr=smo2_errors,fmt='*-', capsize=10, capthick=2,elinewidth=3,ms=10,linewidth=2)
axes[1,1].errorbar(x, smo4, label='SPQ', linestyle=':', color='dodgerblue', yerr=smo4_errors,fmt='s-', capsize=10, capthick=2,elinewidth=3,ms=10,linewidth=3)
axes[1,1].errorbar(x, smo3, label='QoE-TILE', linestyle='-.', color='sandybrown', yerr=smo3_errors,fmt='p-', capsize=10, capthick=2,elinewidth=3,ms=10,linewidth=2)
axes[1,1].set_xlabel('Network Index',fontsize=24,labelpad=14)
axes[1,1].set_ylabel('Quality variation',fontsize=24)
axes[1,1].xaxis.set_tick_params(labelsize=24)
axes[1,1].yaxis.set_tick_params(labelsize=24)
axes[1,1].grid(True)
axes[1,1].set_yticks([0.6,0.8,1.0])
#axes[1,1].set_xticks(np.arange(1, 4, 1)) 
# axes[1,1].set_xlim([0, 1.5])
# axes[1,1].set_ylim([0, 1])
# 添加图例
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.53, 0.98), ncol=4, fontsize=24)

plt.tight_layout(rect=[0, 0, 1, 0.9])  # 调整布局以防止图例与子图重叠
plt.subplots_adjust(wspace =0.29, hspace =0.25)
plt.savefig('wa.svg', dpi=600, orientation='portrait', format='svg',transparent=False, bbox_inches='tight', pad_inches=0.1)
plt.show()