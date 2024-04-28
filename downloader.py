import os
import env
import time
current_directory = str(os.path.dirname(os.path.realpath(__file__)))
dl_gof=0
player_frame=-1
buffer=[]
dl_time=0
for i in range(10):
    buffer.append([])
    for j in range(12):
        buffer[i].append(-1)
while 1:
    if os.path.exists(current_directory+'/downloading/ready.txt'):
        break
while 1:
    if player_frame>=270:
        break
    while 1:
        with open(current_directory+'/downloading/player_frame.txt', 'r') as file:
            line=file.readline()
            if line!='':
                player_frame=int(line) 
                break   
    play_gof=player_frame//30
    for dl_gof in range(play_gof+1,10):
        pred_tile=env.fov_pred(player_frame,dl_gof)
        tile_need=[0]*12
        for t in range(12):
            if buffer[dl_gof][t]==-1 and  pred_tile[t]==1:
                tile_need[t]=1
        if sum(tile_need)>0:
            with open(current_directory+'/downloading/dl_gof.txt', 'w') as file:
                file.write(str(dl_gof))
            cost_time,dl_level=env.download(dl_gof,tile_need,player_frame)      
            for t in range(12):
                if buffer[dl_gof][t]==-1 and  pred_tile[t]==1:
                    buffer[dl_gof][t]=dl_level[t]
            dl_time+=1
            with open(current_directory+'/downloading/dl_'+str(dl_time)+'.txt', 'w') as file:
                file.write(str(dl_gof)+'\n')
                for t in range(12):
                    if tile_need[t]==1:
                        file.write(str(t)+' '+str(dl_level[t])+'\n')
            with open(current_directory+'/downloading/dl_time.txt', 'w') as file:
                file.write(str(dl_time))  
            print(dl_time,player_frame,dl_gof,cost_time)
            time.sleep(cost_time)
            with open(current_directory+'/downloading/dl_gof.txt', 'w') as file:
                file.write(str(999)) 
            print(buffer)  
            break        
                        
        if dl_gof==9:
            time.sleep(0.1) 
            break      
 