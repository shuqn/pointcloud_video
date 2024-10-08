import pandas as pd
import open3d as o3d
import csv
import os
import numpy as np
TILE_DIVISION=[2,3,2]
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
# with open('e:/code/Volumetric-Video-Streaming/fov_model/Video_1.csv', 'r') as file:
#     reader = csv.reader(file)
#     rows = list(reader)
#     for i in range(9001,12001):
#         pcd = o3d.io.read_point_cloud('e:/code/Volumetric-Video-Streaming/model3/data/Videos/Video_1/longdress_vox10_'+str(1050+int(rows[i][0]))+'.ply')
#         fov,dis=make_fov_tiles(pcd,list(map(float, rows[i][1:7])))
#         data_to_insert = {'user': rows[i][7][1:], 'frame': rows[i][0]} 
#         for c in range(12):
#             data_to_insert['r'+str(c)]=fov[c]
#         real_df=pd.concat([real_df,pd.DataFrame(data_to_insert, index=[0])], ignore_index=True)
#         real_df.to_csv('e:/code/Volumetric-Video-Streaming/fov_model/real_tile.csv', index=False)
# with open('e:/code/Volumetric-Video-Streaming/fov_model/prediction.csv', 'r') as file:
# # 创建CSV读取器
#     reader = csv.reader(file)
#     rows = list(reader)
#     for i in range(1,144001):#改这里
#         pcd = o3d.io.read_point_cloud('e:/code/Volumetric-Video-Streaming/model3/data/Videos/Video_1/longdress_vox10_'+str(1050+int(rows[i][1]))+'.ply')
#         fov,dis=make_fov_tiles(pcd,list(map(float, rows[i][3:9])))
#         data_to_insert = {'user': rows[i][0], 'frame': rows[i][1], 'window': rows[i][2]} 
#         for c in range(12):
#             data_to_insert[str(c)]=fov[c]
#         df=pd.concat([df,pd.DataFrame(data_to_insert, index=[0])], ignore_index=True)
#     df.to_csv('e:/code/Volumetric-Video-Streaming/fov_model/prediction_tile.csv', index=False)

df1 = pd.read_csv('e:/code/Volumetric-Video-Streaming/fov_model/prediction_tile.csv', index_col=None)
df2 = pd.read_csv('e:/code/Volumetric-Video-Streaming/fov_model/real_tile.csv', index_col=None)
# df1 = df1.drop('12', axis=1)
df2 = df2.drop('r12', axis=1)
print(df1)
df2=df2[df2['user'] >= 31]
df2=df2[df2['frame'] >= 51]
# 进行左外连接
result = pd.merge(df1, df2, on=['user','frame'], how='left')
# print(result)
# result.to_csv('e:/code/Volumetric-Video-Streaming/fov_model/join.csv')
precision=[]
recall=[]
peidiction_window=90
for i in range(peidiction_window):
    precision.append([])
    recall.append([])
for index, row in result.iterrows():
    same=0
    real=0
    pred=0
    for i in range(12):
        if(row[str(i)]==1):
            pred+=1
        if (row['r'+str(i)]==1):
            real+=1
        if (row[str(i)]==1)and(row['r'+str(i)]==1):
            same+=1
    if pred==0:
        precision[row['window']].append(1)
    else:
        precision[row['window']].append(same/pred)
    if real==0:
        recall[row['window']].append(1)
    else:
        recall[row['window']].append(same/real)
mean_precision = [sum(sublist) / len(sublist) for sublist in precision]
mean_recall = [sum(sublist) / len(sublist) for sublist in recall]
print(mean_precision)
print(mean_recall)
# [0.9964870817013677, 0.9946459664673953, 0.9932814024599741, 0.9911224146224149, 0.988591630591631, 0.9864038857967431, 0.9844838349481208, 0.982163196591768, 0.9798588778945919, 0.9772999381570809, 0.9742345564488418, 0.9704682711468423, 0.966298907441764, 0.9628862949220086, 0.9601813543599255, 0.9575795884010168, 0.9532656496942213, 0.9500966467395049, 0.9483618154332457, 0.946273122380268, 0.9445640589569192, 0.9425408335051225, 0.9411625781625815, 0.9392055418126882, 0.9377186662543838, 0.9367238541881432, 0.9357998694427299, 0.9347380952380983, 0.9340358345358374, 0.9327411530268703, 0.9317074314574343, 0.930290661719236, 0.9291369477083792, 0.9280204253418567, 0.9271193568336452, 0.926624888339177, 0.9262927059712802, 0.9264827011612754, 0.9266052360338103, 0.9261001855287597]
# [0.9975404040404044, 0.9958763828763832, 0.9942917267917273, 0.9919296536796541, 0.9887048546691406, 0.986390194461623, 0.9844599223527792, 0.9819854325568602, 0.9802858860715993, 0.9779651961794813, 0.9754282793925642, 0.9730393217893211, 0.9716411221054072, 0.9700247028104165, 0.9670944135229849, 0.9641511028653886, 0.9619783549783552, 0.9601128117913831, 0.9579570707070705, 0.9560364014292583, 0.9533838040266609, 0.9517256751185321, 0.9491588160516734, 0.9470561224489799, 0.9451126915412634, 0.9436985844842994, 0.9422120524977674, 0.9410139146567724, 0.9388624853982, 0.9365703463203463, 0.9347846663918095, 0.932882790489934, 0.930476379440666, 0.9283990242561683, 0.9267484367484377, 0.9259881983096278, 0.923923761423762, 0.9216429945715661, 0.9197928605785755, 0.9170353191781773]

# [0.9959847357503605, 0.9944041756854256, 0.9932279265873015, 0.9916704094516594, 0.989811755952381, 0.9869624368686868, 0.9853121843434345, 0.9827315340909094, 0.9807533820346325, 0.9780305284992792, 0.9748206168831178, 0.9715397952741711, 0.9690115440115448, 0.9641625856782113, 0.9597762671356429, 0.9578974792568551, 0.956882801226552, 0.9555428616522373, 0.954536525974026, 0.9539969336219333, 0.9535189168470413, 0.9528124549062043, 0.9519636318542561, 0.9516168154761898, 0.9510728941197685, 0.950577899531024, 0.9502053796897542, 0.950161142676767, 0.9499786931818176, 0.9502960858585853, 0.9508001893939386, 0.9514851641414136, 0.9515465593434334, 0.9517215458152948, 0.9519197781385271, 0.9514939123376617, 0.9510246888528131, 0.9509458423520917, 0.9510773133116877, 0.9507079951298695, 0.950578688672438, 0.9503482593795087, 0.950190543831168, 0.9500674377705619, 0.9502053796897539, 0.949841472763347, 0.9496429698773443, 0.9492419056637803, 0.9493508973665221, 0.9492990845959594, 0.9494037472943723, 0.9493874458874457, 0.9496729797979795, 0.9494239267676763, 0.9493374368686864, 0.9494479166666663, 0.9498901515151512, 0.9499857954545452, 0.9500713383838381, 0.9501134785353532, 0.9501032196969696, 0.9501685606060606, 0.9500074179292929, 0.9499952651515151, 0.9499350198412698, 0.9497350514069263, 0.9493613140331886, 0.9492810921717166, 0.9494397546897543, 0.9492970328282822, 0.9496865530303025, 0.9495969065656559, 0.9499491792929282, 0.9500255681818172, 0.9500710227272718, 0.9499914772727263, 0.9499154040404031, 0.950356376262626, 0.9509573863636366, 0.9510809659090913, 0.9510883838383843, 0.9515645517676774, 0.9517616792929299, 0.9529059343434346, 0.9531982323232328, 0.9537009154040408, 0.9541352588383838, 0.9542220643939392, 0.9541090593434339, 0.9541410984848478]
# [0.9973342803030305, 0.996028724747475, 0.9944662247474748, 0.9908508973665225, 0.9883957656926405, 0.9858699720418466, 0.9821353490259734, 0.9800279581529575, 0.9776462617243863, 0.9752221545815288, 0.972722154581529, 0.9684309613997111, 0.9656300279581531, 0.9620449359668116, 0.9599028228715736, 0.9576002435064943, 0.9549675099206355, 0.9519273313492067, 0.9488841991341996, 0.9461612103174609, 0.9443008432539689, 0.9426443903318915, 0.9396659677128436, 0.9372100694444447, 0.9345326253607502, 0.9273451253607499, 0.9236169056637794, 0.9203956755050495, 0.9133011814574306, 0.9048411120129856, 0.8967161120129855, 0.8897344200937937, 0.8791268262986993, 0.8723372564935041, 0.8656321473665202, 0.8575339781745996, 0.8499055059523773, 0.8434337572150042, 0.8370275072150046, 0.8322050640331871, 0.8292308802308785, 0.8258144615800853, 0.8207737644300134, 0.8164987599206343, 0.8123829139610388, 0.8090200667388174, 0.8066065566378076, 0.8033309884559894, 0.7999889745671009, 0.7959160579004342, 0.7929993912337672, 0.7899376803751805, 0.78641957521645, 0.7836070752164498, 0.7792714871933613, 0.7758166260822502, 0.7709828192640679, 0.7663548656204894, 0.7615555104617593, 0.7578905122655112, 0.7547775974025963, 0.7491527552308793, 0.7457173971861465, 0.7401506132756122, 0.735699697871571, 0.7323202110389583, 0.7292386138167357, 0.7252573953823925, 0.722118709415582, 0.7200911345598823, 0.7160204274891752, 0.7134841946248173, 0.711439980158728, 0.7075579680735907, 0.7035874143217878, 0.6966545589826832, 0.6923170093795088, 0.686878517316017, 0.6823554518398267, 0.6769571383477626, 0.6723137175324669, 0.665570391414142, 0.6594319534632034, 0.6550130546536782, 0.6508672213203441, 0.6467578463203434, 0.6422041847041828, 0.6370402011183248, 0.6317303841991332, 0.6259902371933613]