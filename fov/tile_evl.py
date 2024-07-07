import pandas as pd
import open3d as o3d
import csv
import os
import numpy as np
TILE_DIVISION=[2,3,2]
import os
current_directory = str(os.path.dirname(os.path.realpath(__file__)))
def make_fov_tiles(pcd,fov):
    # !!!注意，使用的fov数据集要求点云绕y旋转90°，并且缩放为0.0018！！！左手系右手系
    points_np = np.asarray(pcd.points)
    points_np[:, 2] *= -1

    # 将修改后的NumPy数组重新赋值给点云
    pcd.points = o3d.utility.Vector3dVector(points_np)
    scaling_factor = 0.0018  # 缩放为0.1倍
    pcd.scale(scaling_factor,np.array([0, 0, 0]))

    # 创建变换矩阵
    rotation_angle = -np.pi / 2  # 90 度

    # 定义绕 Y 轴的旋转矩阵
    rotation_matrix = np.array([
        [np.cos(rotation_angle), 0, np.sin(rotation_angle)],
        [0, 1, 0],
        [-np.sin(rotation_angle), 0, np.cos(rotation_angle)]
    ])

    # 对点云进行旋转
    pcd.rotate(rotation_matrix, center=(0, 0, 0))
    central_points = [[0, 0, 0]]
    lookat_points = [[0, 0, -1]]

    # 创建指向点云
    point_cloud_central = o3d.geometry.PointCloud()
    point_cloud_central.points = o3d.utility.Vector3dVector(np.array(central_points))
    point_cloud_lookat = o3d.geometry.PointCloud()
    point_cloud_lookat.points = o3d.utility.Vector3dVector(np.array(lookat_points))
    
    rotation_center = np.array([0, 0, 0])
    rotation_angles = np.radians([fov[3], -fov[4],-fov[5]])  # 替换为你想要的旋转角度

    # 获取旋转矩阵
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(rotation_angles)
    # 对点云应用旋转矩阵
    point_cloud_central.rotate(rotation_matrix, rotation_center)

    translation_vector = np.array([fov[0], fov[1],-fov[2]])
    point_cloud_central.translate(translation_vector)
    point_cloud_lookat.rotate(rotation_matrix, rotation_center)

    translation_vector = np.array([fov[0], fov[1],-fov[2]])
    point_cloud_lookat.translate(translation_vector)


    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()

    x_step = (max_bound[0] - min_bound[0]) / TILE_DIVISION[0]
    y_step = (max_bound[1] - min_bound[1]) / TILE_DIVISION[1]
    z_step = (max_bound[2] - min_bound[2]) / TILE_DIVISION[2]

    tiles_fov = []
    tiles_dis=[]
    for i in range(TILE_DIVISION[0]):
        for j in range(TILE_DIVISION[1]):
            for k in range(TILE_DIVISION[2]):
                min_point = [
                    min_bound[0] + i * x_step,
                    min_bound[1] + j * y_step,
                    min_bound[2] + k * z_step
                ]
                max_point = [
                    min_bound[0] + (i+1) * x_step,
                    min_bound[1] + (j+1) * y_step,
                    min_bound[2] + (k+1) * z_step
                ]
                
                vertices = [
                    min_point,
                    [max_point[0], min_point[1], min_point[2]],
                    [min_point[0], max_point[1], min_point[2]],
                    [min_point[0], min_point[1], max_point[2]],
                    [max_point[0], max_point[1], min_point[2]],
                    [max_point[0], min_point[1], max_point[2]],
                    [min_point[0], max_point[1], max_point[2]],
                    max_point
                ]
                in_fov=0
                distance=np.linalg.norm((np.array(min_point)+np.array(max_point))/2-np.asarray(point_cloud_central.points).flatten())
                for vert in vertices:
                    v1 = (np.array(vert)-np.asarray(point_cloud_central.points)).flatten()
                    v2 = (np.asarray(point_cloud_lookat.points)-np.asarray(point_cloud_central.points)).flatten()
                    deg=np.degrees(np.arccos(np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2)))
                    # print(deg)
                    if deg<=30:
                        in_fov=1
                        break
                tiles_dis.append(distance)
                tiles_fov.append(in_fov)
    return tiles_fov,tiles_dis

df = pd.DataFrame(columns=['user', 'frame', 'window','0','1','2','3','4','5','6','7','8','9','10','11'])
# real_df=pd.DataFrame(columns=['user', 'frame','r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11'])
# with open(current_directory+'/Video_1.csv', 'r') as file:
#     reader = csv.reader(file)
#     rows = list(reader)
#     for i in range(9001,12001):
#         pcd = o3d.io.read_point_cloud('e:/code/Volumetric-Video-Streaming/model2/data/Videos/Video_1/longdress_vox10_'+str(1050+int(rows[i][0]))+'.ply')
#         fov,dis=make_fov_tiles(pcd,list(map(float, rows[i][1:7])))
#         data_to_insert = {'user': rows[i][7][1:], 'frame': rows[i][0]} 
#         for c in range(12):
#             data_to_insert['r'+str(c)]=fov[c]
#         real_df=pd.concat([real_df,pd.DataFrame(data_to_insert, index=[0])], ignore_index=True)
#         real_df.to_csv(current_directory+'/real_tile.csv', index=False)
# with open(current_directory+'/prediction.csv', 'r') as file:
# # 创建CSV读取器
#     reader = csv.reader(file)
#     rows = list(reader)
#     for i in range(1,144001):#改这里
#         pcd = o3d.io.read_point_cloud('e:/code/Volumetric-Video-Streaming/model2/data/Videos/Video_1/longdress_vox10_'+str(1050+int(rows[i][1]))+'.ply')
#         fov,dis=make_fov_tiles(pcd,list(map(float, rows[i][3:9])))
#         data_to_insert = {'user': rows[i][0], 'frame': rows[i][1], 'window': rows[i][2]} 
#         for c in range(12):
#             data_to_insert[str(c)]=fov[c]
#         df=pd.concat([df,pd.DataFrame(data_to_insert, index=[0])], ignore_index=True)
#     df.to_csv(current_directory+'/prediction_tile.csv', index=False)

df1 = pd.read_csv(current_directory+'/prediction_tile.csv', index_col=None)
df2 = pd.read_csv(current_directory+'/real_tile.csv', index_col=None)
# df1 = df1.drop('12', axis=1)
df2 = df2.drop('r12', axis=1)
print(df1)
df2=df2[df2['user'] >= 31]
df2=df2[df2['frame'] >= 51]
# 进行左外连接
result = pd.merge(df1, df2, on=['user','frame'], how='left')
# print(result)
# result.to_csv(current_directory+'/join.csv')
precision=[]
recall=[]
acc=[]
peidiction_window=90
for i in range(peidiction_window):
    precision.append([])
    recall.append([])
    acc.append([])
for index, row in result.iterrows():
    same=0
    real=0
    pred=0
    ac=0
    for i in range(12):
        if(row[str(i)]==1):
            pred+=1
        if (row['r'+str(i)]==1):
            real+=1
        if (row[str(i)]==1)and(row['r'+str(i)]==1):
            same+=1
        if row[str(i)]==row['r'+str(i)]:
            ac+=1
    acc[row['window']].append(ac/12)
    if pred==0:
        precision[row['window']].append(1)
    else:
        precision[row['window']].append(same/pred)
    if real==0:
        recall[row['window']].append(1)
    else:
        recall[row['window']].append(same/real)
meac_acc=[sum(sublist) / len(sublist) for sublist in acc]
mean_precision = [sum(sublist) / len(sublist) for sublist in precision]
mean_recall = [sum(sublist) / len(sublist) for sublist in recall]
print(meac_acc)
print(mean_precision)
print(mean_recall)
# [0.9954166666666666, 0.9935937500000002, 0.9913541666666665, 0.9879687499999998, 0.9848958333333332, 0.9810937499999999, 0.9768750000000004, 0.9734375000000007, 0.9698437500000004, 0.9662500000000006, 0.9620312500000011, 0.9563020833333346, 0.9517187500000017, 0.945833333333335, 0.9409895833333348, 0.9360416666666683, 0.9313541666666678, 0.9258854166666677, 0.9209375000000013, 0.9155729166666675, 0.9125520833333344, 0.9098437500000013, 0.9058854166666673, 0.9026562499999998, 0.8990104166666665, 0.8913541666666659, 0.8880208333333318, 0.8852083333333323, 0.8785937499999993, 0.8713541666666653, 0.864947916666665, 0.8594791666666646, 0.8505729166666635, 0.8448958333333301, 0.8390104166666633, 0.8318749999999956, 0.8250520833333289, 0.8190624999999957, 0.8133854166666629, 0.808906249999997, 0.8059374999999973, 0.8023437499999977, 0.7977604166666645, 0.7936979166666646, 0.7905208333333316, 0.7873958333333324, 0.7852083333333324, 0.7822395833333321, 0.7795312499999993, 0.7764583333333325, 0.7742708333333322, 0.7718749999999988, 0.7692708333333319, 0.7667708333333317, 0.7629687499999975, 0.7599479166666642, 0.7561458333333307, 0.7522916666666641, 0.7483333333333309, 0.7452604166666645, 0.7426562499999979, 0.737812499999998, 0.7346874999999982, 0.7301562499999978, 0.7265104166666638, 0.7237499999999968, 0.7207812499999967, 0.7171874999999973, 0.7145833333333313, 0.7124479166666648, 0.709322916666665, 0.7070312499999983, 0.7055208333333317, 0.7022916666666653, 0.6989583333333327, 0.6927604166666661, 0.6884374999999997, 0.6836979166666662, 0.6797916666666656, 0.6747395833333311, 0.6704166666666646, 0.6648958333333316, 0.659635416666664, 0.6569791666666633, 0.6539062499999959, 0.6504687499999952, 0.6468229166666616, 0.6424999999999962, 0.6377083333333313, 0.6331770833333321]
# [0.9959847357503605, 0.9944041756854256, 0.9932279265873015, 0.9916704094516594, 0.989811755952381, 0.9869624368686868, 0.9853121843434345, 0.9827315340909094, 0.9807533820346325, 0.9780305284992792, 0.9748206168831178, 0.9715397952741711, 0.9690115440115448, 0.9641625856782113, 0.9597762671356429, 0.9578974792568551, 0.956882801226552, 0.9555428616522373, 0.954536525974026, 0.9539969336219333, 0.9535189168470413, 0.9528124549062043, 0.9519636318542561, 0.9516168154761898, 0.9510728941197685, 0.950577899531024, 0.9502053796897542, 0.950161142676767, 0.9499786931818176, 0.9502960858585853, 0.9508001893939386, 0.9514851641414136, 0.9515465593434334, 0.9517215458152948, 0.9519197781385271, 0.9514939123376617, 0.9510246888528131, 0.9509458423520917, 0.9510773133116877, 0.9507079951298695, 0.950578688672438, 0.9503482593795087, 0.950190543831168, 0.9500674377705619, 0.9502053796897539, 0.949841472763347, 0.9496429698773443, 0.9492419056637803, 0.9493508973665221, 0.9492990845959594, 0.9494037472943723, 0.9493874458874457, 0.9496729797979795, 0.9494239267676763, 0.9493374368686864, 0.9494479166666663, 0.9498901515151512, 0.9499857954545452, 0.9500713383838381, 0.9501134785353532, 0.9501032196969696, 0.9501685606060606, 0.9500074179292929, 0.9499952651515151, 0.9499350198412698, 0.9497350514069263, 0.9493613140331886, 0.9492810921717166, 0.9494397546897543, 0.9492970328282822, 0.9496865530303025, 0.9495969065656559, 0.9499491792929282, 0.9500255681818172, 0.9500710227272718, 0.9499914772727263, 0.9499154040404031, 0.950356376262626, 0.9509573863636366, 0.9510809659090913, 0.9510883838383843, 0.9515645517676774, 0.9517616792929299, 0.9529059343434346, 0.9531982323232328, 0.9537009154040408, 0.9541352588383838, 0.9542220643939392, 0.9541090593434339, 0.9541410984848478]
# [0.9973342803030305, 0.996028724747475, 0.9944662247474748, 0.9908508973665225, 0.9883957656926405, 0.9858699720418466, 0.9821353490259734, 0.9800279581529575, 0.9776462617243863, 0.9752221545815288, 0.972722154581529, 0.9684309613997111, 0.9656300279581531, 0.9620449359668116, 0.9599028228715736, 0.9576002435064943, 0.9549675099206355, 0.9519273313492067, 0.9488841991341996, 0.9461612103174609, 0.9443008432539689, 0.9426443903318915, 0.9396659677128436, 0.9372100694444447, 0.9345326253607502, 0.9273451253607499, 0.9236169056637794, 0.9203956755050495, 0.9133011814574306, 0.9048411120129856, 0.8967161120129855, 0.8897344200937937, 0.8791268262986993, 0.8723372564935041, 0.8656321473665202, 0.8575339781745996, 0.8499055059523773, 0.8434337572150042, 0.8370275072150046, 0.8322050640331871, 0.8292308802308785, 0.8258144615800853, 0.8207737644300134, 0.8164987599206343, 0.8123829139610388, 0.8090200667388174, 0.8066065566378076, 0.8033309884559894, 0.7999889745671009, 0.7959160579004342, 0.7929993912337672, 0.7899376803751805, 0.78641957521645, 0.7836070752164498, 0.7792714871933613, 0.7758166260822502, 0.7709828192640679, 0.7663548656204894, 0.7615555104617593, 0.7578905122655112, 0.7547775974025963, 0.7491527552308793, 0.7457173971861465, 0.7401506132756122, 0.735699697871571, 0.7323202110389583, 0.7292386138167357, 0.7252573953823925, 0.722118709415582, 0.7200911345598823, 0.7160204274891752, 0.7134841946248173, 0.711439980158728, 0.7075579680735907, 0.7035874143217878, 0.6966545589826832, 0.6923170093795088, 0.686878517316017, 0.6823554518398267, 0.6769571383477626, 0.6723137175324669, 0.665570391414142, 0.6594319534632034, 0.6550130546536782, 0.6508672213203441, 0.6467578463203434, 0.6422041847041828, 0.6370402011183248, 0.6317303841991332, 0.6259902371933613]