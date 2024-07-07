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
real_df=pd.DataFrame(columns=['user', 'frame','r0','r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11'])
# with open(current_directory+'/Video_1.csv', 'r') as file:
#     reader = csv.reader(file)
#     rows = list(reader)
#     for i in range(9001,12001):
#         pcd = o3d.io.read_point_cloud(current_directory+'/data/Videos/Video_1/longdress_vox10_'+str(1050+int(rows[i][0]))+'.ply')
#         fov,dis=make_fov_tiles(pcd,list(map(float, rows[i][1:7])))
#         data_to_insert = {'user': rows[i][7][1:], 'frame': rows[i][0]} 
#         for c in range(12):
#             data_to_insert['r'+str(c)]=fov[c]
#         real_df=pd.concat([real_df,pd.DataFrame(data_to_insert, index=[0])], ignore_index=True)
#         real_df.to_csv(current_directory+'/real_tile.csv', index=False)
# with open(current_directory+'/mlp_prediction.csv', 'r') as file:
# # 创建CSV读取器
#     reader = csv.reader(file)
#     rows = list(reader)
#     for i in range(1,144001):#改这里
#         pcd = o3d.io.read_point_cloud(current_directory+'/data/Videos/Video_1/longdress_vox10_'+str(1050+int(rows[i][1]))+'.ply')
#         fov,dis=make_fov_tiles(pcd,list(map(float, rows[i][3:9])))
#         data_to_insert = {'user': rows[i][0], 'frame': rows[i][1], 'window': rows[i][2]} 
#         for c in range(12):
#             data_to_insert[str(c)]=fov[c]
#         df=pd.concat([df,pd.DataFrame(data_to_insert, index=[0])], ignore_index=True)
#     df.to_csv(current_directory+'/mlp_prediction_tile.csv', index=False)

df1 = pd.read_csv(current_directory+'/mlp_prediction_tile.csv', index_col=None)
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
# [0.9956250000000001, 0.9935937500000003, 0.9906770833333336, 0.9876041666666666, 0.9854166666666663, 0.9820312499999995, 0.9780208333333323, 0.9749999999999992, 0.9699479166666661, 0.964791666666666, 0.9603645833333329, 0.9559895833333338, 0.9503645833333344, 0.9442708333333353, 0.9384895833333351, 0.9325520833333353, 0.9270833333333346, 0.9201562500000015, 0.9113020833333354, 0.9031770833333356, 0.8952083333333343, 0.8857812499999992, 0.8745312499999988, 0.8644270833333328, 0.8503124999999998, 0.8358854166666663, 0.8240624999999998, 0.8067187500000003, 0.7915624999999994, 0.7781770833333327, 0.7624999999999988, 0.7457291666666648, 0.7304687499999981, 0.7192708333333322, 0.7089583333333321, 0.7016666666666661, 0.6963020833333324, 0.6913541666666662, 0.685677083333333, 0.681093749999999, 0.679218749999999, 0.677083333333332, 0.6721874999999984, 0.6708854166666647, 0.6716145833333313, 0.6769270833333313, 0.6814062499999973, 0.6809895833333306, 0.6834895833333305, 0.6884374999999967, 0.6908333333333299, 0.6925520833333298, 0.6940624999999967, 0.691614583333331, 0.687031249999999, 0.6814583333333332, 0.6760416666666673, 0.6710416666666681, 0.6661979166666679, 0.6620833333333338, 0.656979166666666, 0.6524999999999993, 0.6464062499999984, 0.639687499999998, 0.6322916666666645, 0.6260937499999973, 0.6204687499999973, 0.6170312499999974, 0.613072916666664, 0.6094270833333301, 0.6059374999999969, 0.6014062499999963, 0.5993749999999963, 0.5974999999999963, 0.5963541666666631, 0.5943749999999965, 0.5931770833333297, 0.5910937499999961, 0.5883854166666626, 0.585885416666663, 0.5848958333333305, 0.582916666666664, 0.580833333333331, 0.5794270833333311, 0.5772395833333313, 0.5751562499999978, 0.5739583333333317, 0.5722916666666649, 0.5715104166666651, 0.5696354166666653]
# [0.9963322736291487, 0.9942775072150073, 0.992325983044733, 0.9904262040043293, 0.9881017992424248, 0.9860071473665226, 0.9838567370129873, 0.9818972988816742, 0.9795222988816742, 0.976113929473305, 0.9727521870490625, 0.9694227317821068, 0.9660215322871576, 0.9622879013347767, 0.9601078869047622, 0.9582407557720064, 0.9574072420634924, 0.9565395021645026, 0.955256944444445, 0.9538153634559889, 0.9531866432178938, 0.951602813852815, 0.9508366026334787, 0.9499304428210686, 0.9491617288961045, 0.9473505817099575, 0.9461177173520933, 0.9449965052308812, 0.9433971861471872, 0.9433468389249647, 0.9431652687590193, 0.9429017631673886, 0.9434590548340551, 0.942250586219336, 0.9420815070346319, 0.9420611020923516, 0.9428772546897544, 0.9426957296176041, 0.9437162923881668, 0.9436841856060603, 0.9434734172077919, 0.9431437590187588, 0.9426026334776332, 0.942763618326118, 0.9426929112554108, 0.943248421717171, 0.9437247023809517, 0.9440852498196244, 0.9438362193362189, 0.9434595959595952, 0.9432238455988449, 0.943237847222221, 0.9429790764790752, 0.9432174648268385, 0.9430368867243852, 0.9434710948773436, 0.9439799558080796, 0.9449227768759008, 0.9464431141774879, 0.9479928075396813, 0.9486439168470411, 0.9486916486291476, 0.9490864222582961, 0.9491371527777767, 0.9490368190836928, 0.9490611246392484, 0.9496506358225097, 0.949874458874458, 0.9494051001082241, 0.9489477588383828, 0.9486458784271273, 0.9483582927489167, 0.9486257891414133, 0.949340503246752, 0.9502755907287146, 0.94979024621212, 0.9501788419913406, 0.9500584866522357, 0.9500470102813844, 0.9494401605339098, 0.9489333513708504, 0.948633860930735, 0.9478380005411247, 0.9468233225108217, 0.9465848439754682, 0.9459902371933613, 0.9456748286435774, 0.9455567956349195, 0.9451966991341978, 0.9439507124819612]
# [0.9973012941919194, 0.9961125315656567, 0.9942045454545455, 0.9914172979797978, 0.9900207656926402, 0.9873833874458872, 0.9845773358585853, 0.9820822961760455, 0.9775036976911972, 0.9734771599927847, 0.9707889159451653, 0.9686194534632031, 0.964571518759019, 0.9602636859668117, 0.9553636589105345, 0.9505471681096691, 0.9472641594516604, 0.9445207882395392, 0.938479347041848, 0.9316898674242432, 0.9246382575757579, 0.9172833468614714, 0.9075784406565651, 0.8985573818542558, 0.8848917523448766, 0.8712625586219331, 0.8604322240259744, 0.843833468614719, 0.8294155393217892, 0.8159140512265511, 0.7992092126623374, 0.7808014745670991, 0.7630549693362184, 0.7511258793290034, 0.7394969336219323, 0.7313853715728706, 0.7240759604978352, 0.7178783594877342, 0.7104068813131309, 0.7043669507575748, 0.700967374639249, 0.6971302985209225, 0.6907666621572855, 0.6876833288239519, 0.6868694534632016, 0.6910043740981225, 0.6941833288239516, 0.6917306547619022, 0.6914756718975442, 0.6952125496031725, 0.6969096771284249, 0.6970456123737353, 0.6980055916305897, 0.6943714826839817, 0.6886442099567099, 0.6813320256132765, 0.6734425730519498, 0.6664737779581554, 0.6603161075036093, 0.6550116116522375, 0.6486313807720064, 0.6433792839105338, 0.6361865530303031, 0.6285020743145749, 0.6201034000721506, 0.6127514880952386, 0.6061106150793655, 0.6016176948051952, 0.5963479662698418, 0.5925149936868689, 0.5878453057359309, 0.5819675324675323, 0.5782451073232321, 0.5751146509740256, 0.572139272186147, 0.5694332386363634, 0.5675312049062047, 0.5654395968614715, 0.562636476370851, 0.5601965187590183, 0.5596086985930733, 0.5588314844877341, 0.5569843524531018, 0.5562874053030299, 0.5544473079004324, 0.5528329049422793, 0.5518208648989895, 0.5500248917748913, 0.5487955447330443, 0.547445639430014]