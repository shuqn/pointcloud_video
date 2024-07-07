import matplotlib.pyplot as plt
import numpy as np

qoe1 = [34.06,33.19,28.89]
qoe2 = [32.71,26.64,20.81]
qoe3 = [32.8,31.2,26.01]
qoe4 = [30.05,24.73,21.36]
qoe1_errors =[1.74,1.64,1.99]
qoe2_errors =[1.16,0.81,1.72]
qoe3_errors =[1.98,2.06,1.88]
qoe4_errors =[1.61,1.58,1.28]
# 假设有三组数据

quli1 = [53.19,51.55,51.85]
quli2 = [46.94,46.3,44.96]
quli3 = [57.1,57.81,52.7]
quli4 = [51.65,48.66,42.52]
quli1_errors =[1.06,0.94,1.02]
quli2_errors =[0.82,0.63,0.8]
quli3_errors =[0.97,0.95,0.84]
quli4_errors =[1.02,0.83,0.93]

rebuf1 = [1.846,1.763,2.233]
rebuf2 = [1.346,1.88,2.342]
rebuf3 = [2.359,2.591,2.595]
rebuf4 = [2.09,2.32,2.046]
rebuf1_errors =[0.092,0.089,0.12]
rebuf2_errors =[0.074,0.053,0.125]
rebuf3_errors =[0.13,0.137,0.138]
rebuf4_errors =[0.103,0.112,0.1]

smo1 = [0.67,0.73,0.63]
smo2 = [0.77,0.86,0.73]
smo3 = [0.71,0.7,0.73]
smo4 = [0.7,0.73,0.71]
smo1_errors =[0.02,0.02,0.02]
smo2_errors =[0.02,0.02,0.02]
smo3_errors =[0.02,0.02,0.02]
smo4_errors =[0.01,0.02,0.01]
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
axes[0,0].set_yticks([20,25,30,35]) 
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
axes[1,0].set_xlabel('FoV Prediction Accuracy',fontsize=24,labelpad=14)
axes[1,0].set_ylabel('Rebuffering (sec)',fontsize=24)
axes[1,0].xaxis.set_tick_params(labelsize=24)
axes[1,0].yaxis.set_tick_params(labelsize=24)
axes[1,0].grid(True)
axes[1,0].set_yticks([1.2,2.0,2.8])
#axes[1, 0].set_xticks(np.arange(1, 4, 1)) 
# axes[1,0].set_xlim([0, 10])
# axes[1,0].set_ylim([0, 1])

axes[1,1].errorbar(x, smo1, label='PACE', color='tomato', yerr=smo1_errors,fmt='o-', capsize=10, capthick=2,elinewidth=3,zorder=5,ms=10,linewidth=2)
axes[1,1].errorbar(x, smo2, label='POT-DRL', linestyle='--', color='mediumseagreen', yerr=smo2_errors,fmt='*-', capsize=10, capthick=2,elinewidth=3,ms=10,linewidth=2)
axes[1,1].errorbar(x, smo4, label='SPQ', linestyle=':', color='dodgerblue', yerr=smo4_errors,fmt='s-', capsize=10, capthick=2,elinewidth=3,ms=10,linewidth=3)
axes[1,1].errorbar(x, smo3, label='QoE-TILE', linestyle='-.', color='sandybrown', yerr=smo3_errors,fmt='p-', capsize=10, capthick=2,elinewidth=3,ms=10,linewidth=2)
axes[1,1].set_xlabel('FoV Prediction Accuracy',fontsize=24,labelpad=14)
axes[1,1].set_ylabel('Quality variation',fontsize=24)
axes[1,1].xaxis.set_tick_params(labelsize=24)
axes[1,1].yaxis.set_tick_params(labelsize=24)
axes[1,1].grid(True)
axes[1,1].set_yticks([0.6,0.7,0.8,0.9])
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