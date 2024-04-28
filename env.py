import os
import time
import random
import math
current_directory = str(os.path.dirname(os.path.realpath(__file__)))
bw_list = []
time_list=[]
dl_time=0.0
record_delay=[0]*10
with open(current_directory+'/data/trace_5g.txt', 'r') as file:
    i=0    
    for line in file:
        values =line.strip().split()
        time_list.append(float(values[1]))
        bw_list.append(float(values[0])/1000)
        i+=1
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
video_size = []
with open(current_directory+'/data/tile_size.txt', 'r') as file:
    for line in file:
        video_size.append([])
        row = list(map(float, line.split()))
        for t in range(12):
            video_size[-1].append([])
            for l in [1,1.75,2.5,3.25]:
                video_size[-1][t].append(row[t]*l)
gof_recall=[0]*10
dis = []
with open(current_directory+'/data/dis_1.txt', 'r') as file:
    for line in file:
        dis.append(list(map(float, line.split()))   )
def fov_pred(play_frame,pred_gof):#random here
    global fov_gof_list
    pred_len=pred_gof*30-play_frame
    acc=-pred_len/300+1
    ret=[]
    for t in range(12):
        if fov_gof_list[pred_gof][t]==0:
            ret.append(random.choices(population=[0, 1], weights=[acc,1-acc], k=1)[0])
        else:
            ret.append(random.choices(population=[1, 0], weights=[acc,1-acc], k=1)[0])
    return ret
def bw_pred():
    global dl_time,bw_list
    return (bw_list[math.floor(dl_time)]+bw_list[math.ceil(dl_time)])/2
def download(dl_gof,tile_need,play_frame):
    global dl_time,gof_recall,dis,video_size,bw_list,record_delay
    pred_len=dl_gof*30-play_frame
    recall=-pred_len/300+1
    recall_add=recall-gof_recall[dl_gof]
    bit_constrain=bw_pred()*1*recall_add
    min_size=0
    bitrate=[0]*12
    for i in range(30):
        for t in range(12):
            if tile_need[t]:
                min_size+=video_size[dl_gof*30+i][t][0]
    if min_size<bit_constrain:
        cur_dis=dis[dl_gof*30]
        sorted_dis=sorted(cur_dis)
        index_dis = [cur_dis.index(x) for x in sorted_dis]
        for cur_l in range(3, 0, -1):
            if min_size>=bit_constrain:
                break
            for tile in index_dis:
                add_bit=0
                sub_bit=0
                for f in range(30):
                    add_bit+=video_size[dl_gof*30+f][tile][cur_l]
                    sub_bit+=video_size[dl_gof*30+f][tile][0]
                if tile_need[tile] and  min_size+add_bit-sub_bit<=bit_constrain and bitrate[tile]==0:
                    bitrate[tile]=cur_l
                    min_size=min_size+add_bit-sub_bit 
    gof_recall[dl_gof]+=min_size/bw_pred()  
    video_gof_counter_sent=0
    delay=0
    while True:
        throughput = bw_list[math.floor(dl_time)]
        duration = math.ceil(dl_time)-dl_time
        packet_payload = throughput * duration 

        if video_gof_counter_sent + packet_payload > min_size:
            fractional_time=(min_size-video_gof_counter_sent)/throughput
            delay += fractional_time
            dl_time += fractional_time
            break

        video_gof_counter_sent += packet_payload
        delay += duration
        dl_time=math.ceil(dl_time)+0.0001
    record_delay[dl_gof]+=delay
    print(record_delay)
    return delay,bitrate