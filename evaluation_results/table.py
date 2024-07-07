#content	lstm	mlp	tile-dirct

pace_qoe=	[39.92,	39.31,	36.74,	32.9]

pace_quality=	[59.86,	59.1,	57.15,	47.54]

pace_rebuffer=	[1.925,	1.908,	1.964,	1.384]

pace_smooth=	[0.69,	0.72,	0.77,	0.8]

pace_acc = [0.9,0.9,0.89,0.8]



				
seq_qoe=	[35.04,	29,	26.81,	21.25]

seq_quality=	[57.51,	51.48,	50.26,	34.12]

seq_rebuffer=	[2.169,	2.167,	2.275,	1.195]

seq_smooth=	[0.79,	0.81,	0.7,	0.92]

seq_acc= [0.87, 0.83, 0.81, 0.65]


for i in range(4):
	print("QoE")
	print((pace_qoe[i]-seq_qoe[i])/seq_qoe[i])
	
for i in range(4):
	print("quality")
	print((pace_quality[i]-seq_quality[i])/seq_quality[i])
	
for i in range(4):
	print("rebuff")
	print((seq_rebuffer[i]-pace_rebuffer[i])/seq_rebuffer[i])
	
for i in range(4):
	print("smooth")
	print((seq_smooth[i]-pace_smooth[i])/seq_smooth[i])
	
for i in range(4):
	print("acc")
	print((pace_acc[i]-seq_acc[i])/seq_acc[i])