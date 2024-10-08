#经过测试，发现点的数量和文件大小呈现线性正相关关系
#file_size= 0.02217*point_count-162.7  单位kB 
import os
import open3d as o3d
import numpy as np
import Hyperparameters
import csv
TILE_IN_F=Hyperparameters.TILE_IN_F
TILE_DIVISION = Hyperparameters.TILE_DIVISION
current_directory = str(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH=current_directory+'/data'
COOKED_DATA_PATH=current_directory+'/cooked_data'
def make_tiles(video_num):
    # !!!注意，使用的fov数据集要求点云绕y旋转90°，并且缩放为0.0018！！！左手系右手系

    with open(COOKED_DATA_PATH+'/'+'tile_counts_'+str(video_num)+'.txt', 'w') as f:
        for file in os.listdir(DATA_PATH+'/Videos/Video_'+str(video_num)):
            file_path = os.path.join(DATA_PATH+'/Videos/Video_'+str(video_num), file)
            pcd = o3d.io.read_point_cloud(file_path)
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
            # 获取点云的边界
            min_bound = pcd.get_min_bound()
            max_bound = pcd.get_max_bound()

            # 计算每个tile的大小
            tiles_dim = np.array(TILE_DIVISION)  # 你想要的tile维度
            tile_size = (max_bound - min_bound) / tiles_dim

            # 初始化一个用于计数的3D数组
            tile_counts = np.zeros(tiles_dim, dtype=int)

            # 遍历点云中的点并计算它们所属的tile
            points = np.asarray(pcd.points)
            cnt=0
            for point in points:
                cnt+=1
                tile_index = np.floor((point - min_bound) / tile_size).astype(int)
                tile_index = np.clip(tile_index, [0, 0, 0], tiles_dim - 1)  # 防止索引越界
                tile_counts[tuple(tile_index)] += 1
            # for count in tile_counts.flatten():
            for x in range(TILE_DIVISION[0]):
                for y in range(TILE_DIVISION[1]):
                    for z in range(TILE_DIVISION[2]):
                        f.write(str(tile_counts[x,y,z]) + ' ')
            f.flush()
            f.write('\n')

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

# make_fov_tiles(o3d.io.read_point_cloud('e:/1longdress_vox10_1051.ply'),[1.068004,1.578747,0.6945359,13.55903,207.8209,357.843])
def fovs(video_num):
    frames=300
    with open(DATA_PATH+'/Views/Video_'+str(video_num)+'.csv', 'r') as file:
    # 创建CSV读取器
        reader = csv.reader(file)
        rows = list(reader)
        i=0
        for person in range(1,41):
            with open(COOKED_DATA_PATH+'/p'+str(person)+'_fovs_'+str(video_num)+'.txt', 'w') as f:
                with open(COOKED_DATA_PATH+'/p'+str(person)+'_dis_'+str(video_num)+'.txt', 'w') as fd:
                    for file in os.listdir(DATA_PATH+'/Videos/Video_'+str(video_num)):
                        
                        file_path = os.path.join(DATA_PATH+'/Videos/Video_'+str(video_num), file)
                        pcd = o3d.io.read_point_cloud(file_path)
                        fov,dis=make_fov_tiles(pcd,list(map(float, rows[i][1:7])))
                        
                        f.write(' '.join(map(str,fov )) + ' ')
                        f.flush()
                        f.write('\n')
                        fd.write(' '.join(map(str,dis )) + ' ')
                        fd.flush()
                        fd.write('\n')
                        i+=1

def trace_5g():
    import pandas as pd
    df = pd.read_csv(DATA_PATH+'/Traces/5g.csv')
    filtered_data = df[df['DL_bitrate'] > 10000]['DL_bitrate']
    filtered_data.to_csv(COOKED_DATA_PATH+'/trace_5g.txt', index=False, header=False)
    ###插值
    from scipy.interpolate import CubicSpline
    # 读取txt文件
    with open(COOKED_DATA_PATH+'/trace_5g.txt', 'r') as file:
        lines = file.readlines()

    # 将文本行转换为浮点数
    values = [float(line.strip()) for line in lines]

    # 创建插值函数
    x = np.arange(len(values))
    cs = CubicSpline(x, values, bc_type='clamped')  # 使用clamped边界条件

    # 在两个数之间插入九个数
    num_points_between = 9
    new_x = np.linspace(0, len(values) - 1, len(values) + (len(values) - 1) * num_points_between)
    interpolated_values = cs(new_x)

    # 将插值后的结果写回txt文件
    with open(COOKED_DATA_PATH+'/trace_5g_1.txt', 'w') as file:
        for value in interpolated_values:
            file.write(f'{value}\n')
    # 读取txt文件
    with open(COOKED_DATA_PATH+'/trace_5g.txt', 'r') as file:
        lines = file.readlines()

# 处理每一行，为每个数添加相应的行数
    processed_lines = [f'{float(line.strip())} {index}' for index, line in enumerate(lines)]

    # 将处理后的结果写回txt文件
    with open(COOKED_DATA_PATH+'/trace_5g.txt', 'w') as file:
        for processed_line in processed_lines:
            file.write(f'{processed_line}\n')
    # 读取txt文件
    with open(COOKED_DATA_PATH+'/trace_5g_1.txt', 'r') as file:
        lines = file.readlines()

# 处理每一行，为每个数添加相应的行数
    processed_lines = [f'{float(line.strip())} {index*0.1}' for index, line in enumerate(lines)]

    # 将处理后的结果写回txt文件
    with open(COOKED_DATA_PATH+'/trace_5g_1.txt', 'w') as file:
        for processed_line in processed_lines:
            file.write(f'{processed_line}\n')
# make_tiles(1)             
# fovs(1)
trace_5g()
