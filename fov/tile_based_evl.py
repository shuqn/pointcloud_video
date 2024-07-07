import pandas as pd
import open3d as o3d
import csv
import os
import numpy as np
import random
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
def generate_zero_or_one(k):
    if random.random() < k:
        return 1
    else:
        return 0
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


# with open(current_directory+'/real_tile.csv', 'r') as file:
# # 创建CSV读取器
#     reader = csv.reader(file)
#     rows = list(reader)
#     for i in range(1,len(rows)):#改这里
#         if 51<=int(rows[i][1])<=210:
#             data_to_insert = {'user': rows[i][0], 'frame': rows[i][1], 'window':0}
#             for c in range(12):
#                     data_to_insert[str(c)]=int(rows[i][c+2] )
#             for win in range(90):
#                 p_err=0.1+0.3*win/90
#                 data_to_insert['window']=win
#                 for c in range(12):
#                     data_to_insert[str(c)]=generate_zero_or_one((1-2*p_err)*data_to_insert[str(c)]+p_err)
#                 df=pd.concat([df,pd.DataFrame(data_to_insert, index=[0])], ignore_index=True)
#     df.to_csv(current_directory+'/tile_based_prediction_tile.csv', index=False)

df1 = pd.read_csv(current_directory+'/tile_based_prediction_tile.csv', index_col=None)
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
# [0.8958333333333348, 0.8114062499999997, 0.7420833333333323, 0.6896354166666666, 0.6453645833333331, 0.611041666666666, 0.5837499999999994, 0.5613541666666656, 0.5441666666666674, 0.5375000000000008, 0.5272395833333338, 0.5201562500000015, 0.5133333333333324, 0.5089583333333334, 0.5061979166666656, 0.5022395833333327, 0.5023437500000001, 0.5006249999999998, 0.4942708333333335, 0.4954166666666667, 0.49427083333333305, 0.49541666666666745, 0.499635416666666, 0.5003645833333327, 0.49197916666666686, 0.491822916666667, 0.4992187500000007, 0.49859374999999984, 0.5059895833333322, 0.5048437500000001, 0.5003125000000007, 0.5036458333333336, 0.5007812500000002, 0.4995312500000001, 0.49937500000000123, 0.5024479166666669, 0.5047916666666665, 0.5005208333333325, 0.49531250000000004, 0.4978645833333342, 0.49614583333333384, 0.49895833333333256, 0.5044791666666666, 0.502708333333333, 0.5028645833333322, 0.49619791666666513, 0.49536458333333294, 0.49609374999999933, 0.4973437499999992, 0.4948958333333332, 0.49473958333333345, 0.49572916666666633, 0.4989583333333337, 0.5002604166666663, 0.501666666666666, 0.5020833333333339, 0.5005729166666673, 0.4995833333333334, 0.5002083333333331, 0.5020312500000003, 0.5025520833333329, 0.507395833333333, 0.5009375, 0.5008854166666671, 0.4982291666666657, 0.49953125000000015, 0.4995833333333326, 0.5005208333333329, 0.5014583333333336, 0.49442708333333313, 0.5024479166666664, 0.49614583333333323, 0.5002083333333325, 0.4959895833333329, 0.4955208333333334, 0.49817708333333294, 0.48838541666666635, 0.4936458333333333, 0.4964062499999992, 0.49973958333333335, 0.500364583333333, 0.5031770833333329, 0.49786458333333355, 0.49911458333333336, 0.5008854166666666, 0.5027083333333323, 0.5043229166666674, 0.5001041666666662, 0.5017187500000003, 0.5035937499999998]
# [0.9650847988816744, 0.9414639475108233, 0.9202261679292936, 0.9020597492785001, 0.8898590593434349, 0.8794997068903311, 0.8726206259018752, 0.8641121031746027, 0.8572712617243863, 0.8539817821067807, 0.8506045725108211, 0.8480295138888883, 0.8425975378787871, 0.842179991883117, 0.8425851596320336, 0.8391419101731592, 0.8409549963924958, 0.8393400072150066, 0.838892338564213, 0.8412187499999991, 0.8394572736291482, 0.841538194444444, 0.8422041170634919, 0.8440114538239536, 0.8368139880952379, 0.8363308757215, 0.8383030303030296, 0.8362626713564211, 0.8402928165584405, 0.83934061598124, 0.8378333107864347, 0.8380548340548336, 0.8369108270202019, 0.8362283324314567, 0.8392682855339101, 0.8411748737373728, 0.8421700487012974, 0.8395051406926399, 0.8367972808441551, 0.8353670183982678, 0.8367553210678209, 0.8413314393939387, 0.8426341540404028, 0.8422827380952376, 0.8428794417388165, 0.8375143849206341, 0.8405530753968244, 0.8402217938311682, 0.840836467352091, 0.8396867334054829, 0.8356800369769111, 0.836956687409812, 0.8382321428571422, 0.8389577922077911, 0.8425526920995656, 0.8420329861111107, 0.8400829500360742, 0.8393528363997106, 0.8372627164502159, 0.8404215593434337, 0.8385019390331879, 0.8391699810606051, 0.8396796762265509, 0.8426523042929287, 0.8386185064935057, 0.8382718253968252, 0.841039862914862, 0.8429199810606053, 0.8424523809523805, 0.8390832656926404, 0.8374064078282822, 0.8382496167027411, 0.8399176361832608, 0.8374309163059155, 0.8369368235930728, 0.8386630140692637, 0.836105857683982, 0.8398384965728707, 0.8406923250360739, 0.8422762896825384, 0.8428604572510814, 0.8398784722222215, 0.8382256493506485, 0.839997181637806, 0.840877119408369, 0.8433346636002877, 0.8438325667388158, 0.838539547258297, 0.8402915990259728, 0.8416263077200565]
# [0.8979263392857155, 0.813781836219336, 0.7457600559163047, 0.6953097718253968, 0.6518615169552671, 0.6170382169913418, 0.5914746347402585, 0.5684938897907639, 0.5523616973304477, 0.5421979392135644, 0.5341019119769126, 0.5300602227633485, 0.5237657828282825, 0.5162311056998561, 0.5141573097041843, 0.5116270517676771, 0.512890670093795, 0.5098348890692634, 0.502936981421357, 0.5023977949134203, 0.5027240259740259, 0.5048129734848491, 0.5062019976551223, 0.5065226370851366, 0.4985605835137084, 0.5000648899711408, 0.5044217848124106, 0.5034296085858587, 0.5136709280303026, 0.5149779265873018, 0.5098602092352099, 0.5128938266594516, 0.5098851686507937, 0.5078161300505052, 0.5077623331529593, 0.5130404942279947, 0.5160699630230877, 0.511642834595959, 0.5047543064574319, 0.5068900613275621, 0.5052399891774894, 0.5070429969336214, 0.5115907287157284, 0.5124267225829725, 0.5105069444444434, 0.5041658775252513, 0.4989876668470417, 0.5008671987734487, 0.503584641053391, 0.5050757575757578, 0.5061445707070703, 0.5065092667748915, 0.5092364944083696, 0.5135504825036077, 0.5121551001082247, 0.5104719516594527, 0.5071028363997115, 0.5062382981601725, 0.5099982187950933, 0.5089844877344878, 0.5124529220779217, 0.5161983901515146, 0.5091532963564217, 0.5074875090187592, 0.5051776019119766, 0.5097979572510822, 0.5088786751442996, 0.5083164908008657, 0.5077950036075043, 0.5018806592712842, 0.5112968750000002, 0.5043445842352093, 0.51092949585137, 0.5058715954184698, 0.5061809839466086, 0.5029854572510822, 0.49491551677489126, 0.5033501984126982, 0.5025326930014425, 0.5078381132756133, 0.5045808531746033, 0.50876792478355, 0.5034283459595957, 0.508438334235209, 0.510595846861472, 0.5089501939033184, 0.5131564078282832, 0.5062681953463207, 0.5054269705988458, 0.5095165945165945]