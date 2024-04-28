import open3d as o3d
# import open3d.core as o3c
import time
import numpy as np
import os
os.environ['OPEN3D_RENDERING_DEVICE'] = 'cuda'
current_directory = str(os.path.dirname(os.path.realpath(__file__)))
tiles=[]
dl_time=0
for i in range(300):
    tiles.append([])
    for x in range(2):
        for y in range(3):
            for z in range(2):
                key=str((x,y,z))
                tiles[i].append([])
                for l in range(4):                    
                    tiles[i][-1].append(o3d.io.read_point_cloud(current_directory+'/data/longdress/tiles/tile_longdress_vox10_'+str(1051+i)+'.ply'+key+str(-l+7)+".ply"))
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Open3D', width=800, height=600)
render_option = vis.get_render_option()
render_option.point_size = 2.0
fov_list = []
with open(current_directory+'/data/fovs_1.txt', 'r') as file:
    for line in file:
        row = list(map(int, line.split()))
        fov_list.append(row)
fov_gof_list=[]
for g in range(10):
    fov_gof_list.append([0]*12)
    for t in range(12):
        for f in range(30):        
            if fov_list[g*30+f][t]==1:
                fov_gof_list[g][t]=1
buffer=[]
for i in range(10):
    buffer.append([])
    for j in range(12):
        buffer[i].append(-1)
def draw_point_clouds(gof,frame):
    with open(current_directory+'/downloading/player_frame.txt', 'w') as file:
        file.write(str(gof*30+frame))
        file.flush()
    vis.clear_geometries()
    for t in range(12):
        if buffer[gof][t]!=-1:
            vis.add_geometry(tiles[gof*30+frame][t][buffer[gof][t]])
    vis.poll_events()
    vis.update_renderer()
def play_point_cloud_sequence(play_gof):
    start_time = time.time()
    frame_count = 0
    while True:
        draw_point_clouds(play_gof,frame_count)
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > 1.0 / 30:
            start_time = current_time
            frame_count += 1
        if frame_count==29:
            break
def refresh_buffer():
    with open(current_directory+'/downloading/dl_time.txt', 'r') as file:
        max_dl_time=int(file.readline())
    if max_dl_time>dl_time:
        for i in range(dl_time+1,max_dl_time+1):
            with open(current_directory+'/downloading/dl_'+str(i)+'.txt', 'r') as file:
                lines = file.readlines()    
            gof_inx=int(lines[0])
            for line in lines[1:]:
                parts = line.split()
                buffer[gof_inx][int(parts[0])]=int(parts[1])

####
# ready
with open(current_directory+'/downloading/player_frame.txt', 'w') as file:
    file.write('-1')
with open(current_directory+'/downloading/dl_gof.txt', 'w') as file:
    file.write("0")
with open(current_directory+'/downloading/ready.txt', 'w') as file:
    file.write("start!")
    pass
####


play_gof=0
while 1:
    if play_gof==10:
        os.remove(current_directory+'/downloading/ready.txt')
        break
    while 1:
        with open(current_directory+'/downloading/dl_gof.txt', 'r') as file:
            line=file.readline()
            if line!='':
                dl_gof=int(line)
                break
    if dl_gof>play_gof:
        refresh_buffer()
        play_point_cloud_sequence(play_gof)
        play_gof+=1
        
