import matplotlib.pyplot as plt
import numpy as np

# 示例数据
categories = ['Accuracy', 'Recall', 'Precision']
proposed = [0.88, 0.99, 0.87]
pot_drl = [0.84, 0.88, 0.9]
qoe_tile = [0.81, 1, 0.8]
visual_acuity = [0.8, 0.81, 0.9]

# 人工指定的误差棒高度
errors_proposed = [0.03, 0.01, 0.06]
errors_pot_drl = [0.06, 0.02, 0.03]
errors_qoe_tile = [0.06, 0, 0.09]
errors_visual_acuity = [0.06, 0.03, 0.03]

# 柱的宽度和位置
bar_width = 0.2
index = np.arange(len(categories))

# 创建柱状图
fig, ax = plt.subplots(figsize=(12, 3), sharex='none')

# 绘制带有误差棒的柱状图并添加花纹
bars1 = ax.bar(index, proposed, bar_width, label='PACE', yerr=errors_proposed, capsize=10, hatch='//', edgecolor='black', color='tomato')
bars2 = ax.bar(index + bar_width, pot_drl, bar_width, label='POT-DRL', yerr=errors_pot_drl, capsize=10, hatch='\\', edgecolor='black', color='mediumseagreen')
bars4 = ax.bar(index + 2 * bar_width, visual_acuity, bar_width, label='SPQ', yerr=errors_visual_acuity, capsize=10, hatch='||', edgecolor='black', color='skyblue')
bars3 = ax.bar(index + 3 * bar_width, qoe_tile, bar_width, label='QoE-TILE', yerr=errors_qoe_tile, capsize=10, hatch='--', edgecolor='black', color='sandybrown')


# 添加标签和标题
#ax.set_xlabel('Categories')
ax.set_ylabel('Values',fontsize=22)
#ax.set_title('Cell-level evaluation')
ax.set_xticks(index + 1.5 * bar_width)
ax.set_xticklabels(categories)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)
ax.set_yticks([0.6,0.8,1])
ax.set_ylim([0.6, 1])
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.51, 1.2), ncol=4, fontsize=22)
ax.grid(True)

plt.savefig('wa.svg', dpi=600, orientation='portrait', format='svg',transparent=False, bbox_inches='tight', pad_inches=0.1)
# 显示图形
plt.show()
