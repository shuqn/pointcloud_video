import numpy as np
import env_back as env
import load_data
import Hyperparameters
import math
import fov_predict
import os
from Config import Config
current_directory = str(os.path.dirname(os.path.realpath(__file__)))
VIDEO_GOF_LEN=Hyperparameters.VIDEO_GOF_LEN
QUALITY_LEVELS =  Hyperparameters.QUALITY_LEVELS
REBUF_PENALTY = Hyperparameters.REBUF_PENALTY  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = Hyperparameters.SMOOTH_PENALTY
DEFAULT_QUALITY = Hyperparameters.DEFAULT_QUALITY  # default video quality without agent
RANDOM_SEED = Hyperparameters.RANDOM_SEED
RESEVOIR = 1  # BB
CUSHION = 3  # BB
LOG_FILE = current_directory+'/results/log_sim_bw_cons_all'
RESULT_FILE=current_directory+'/results/result_sim_bw_cons_all'
INIT_QOE=Hyperparameters.INIT_QOE
MULTIPLE_QUALITY_LEVELS=Hyperparameters.MULTIPLE_QUALITY_LEVELS
F_IN_GOF=Hyperparameters.F_IN_GOF
TILE_IN_F=Hyperparameters.TILE_IN_F
FRAME=Hyperparameters.FRAME
def test(cooked_time,cooked_bw,video_size,fov,dis,time=None):

    np.random.seed(RANDOM_SEED)

    net_env = env.Environment(cooked_time=cooked_time,
                              cooked_bw=cooked_bw,
                              video_size=video_size
                              )

    log_path = LOG_FILE+str(time)
    log_file = open(log_path, 'w')
    log_file.write('time_stamp' + '\t' +'play_fov' + '\t' +
                       'option' + '\t'+
                       'sum_rebuffer' +  '\t'+
                       'selected_tile' +  '\t'+
                       'bit_rate' + '\n')
    time_stamp = 0

    last_bit_rate = [DEFAULT_QUALITY]*TILE_IN_F
    bit_rate = [DEFAULT_QUALITY]*TILE_IN_F
    player_fov_count=0
    fov_count = 0
    player_fractional_time=0
    sum_reward_quality=0
    sum_reward_rebuffer=0
    sum_reward_switch=0
    #首个gof，全选第一个f的
    selected_tile=[fov[0]]*F_IN_GOF
    selected_tile=[ math.ceil(sum(col) / len(col)) for col in zip(*selected_tile)]
    buffer=[]
    for i in range(FRAME//F_IN_GOF):
        buffer.append([])
        for j in range(TILE_IN_F):
            buffer[i].append(-1)
    delay,buffer=net_env.get_video_gof_new(selected_tile,bit_rate)
    fov_count+=1
    time_stamp += delay
    sum_reward_rebuffer+=delay*REBUF_PENALTY
    log_file.write(str(round(time_stamp,3)) + '\t' + str(player_fov_count+player_fractional_time/VIDEO_GOF_LEN)+ '\t' + 
                    'new0' + '\t'+
                    str(round(sum_reward_rebuffer,1)) + '\t'+ 
                    str(selected_tile) + '\t'+ 
                    str(bit_rate)+'\n')
    while True:      
        log_file.flush()
        back=0
        if fov_count-player_fov_count-player_fractional_time/VIDEO_GOF_LEN>0 and player_fov_count+player_fractional_time>0 and 1==0: #back
            for predict_window in range(player_fov_count+math.ceil(player_fractional_time/VIDEO_GOF_LEN),fov_count):
                predicted_tile=fov_predict.predict(time,int(player_fov_count*F_IN_GOF+player_fractional_time*F_IN_GOF/VIDEO_GOF_LEN),predict_window*F_IN_GOF,(predict_window+1)*F_IN_GOF-1)
                predicted_tile=[ math.ceil(sum(col) / len(col)) for col in zip(*predicted_tile)]
                selected_tile=[0]*TILE_IN_F
                for tile in range(TILE_IN_F):
                    if predicted_tile[tile]==1 and buffer[predict_window][tile]==-1:
                        selected_tile[tile]=1
                if sum(selected_tile)>0:
                    back=1
                    delay,buffer=net_env.get_video_gof_back(predict_window,selected_tile,[0]*TILE_IN_F)
                    time_stamp+=delay
                    if player_fov_count*VIDEO_GOF_LEN+player_fractional_time+delay<predict_window*VIDEO_GOF_LEN: #不rebuffer
                        player_fov_count+=int((player_fractional_time+delay)//VIDEO_GOF_LEN)
                        player_fractional_time=(player_fractional_time+delay)-int((player_fractional_time+delay)//VIDEO_GOF_LEN)*VIDEO_GOF_LEN
                    else:
                        sum_reward_rebuffer+=(player_fov_count*VIDEO_GOF_LEN+player_fractional_time+delay-predict_window*VIDEO_GOF_LEN)*REBUF_PENALTY
                        player_fov_count=predict_window
                        player_fractional_time=0
                    log_file.write(str(round(time_stamp,3)) + '\t' + str(player_fov_count+player_fractional_time/VIDEO_GOF_LEN)+ '\t' + 
                            'back' +str(predict_window)+ '\t'+
                            str(round(sum_reward_rebuffer,1)) + '\t'+ 
                            str(selected_tile) + '\t'+ 
                            str(bit_rate)+'\n')    
                    break
            if back==0 and fov_count==FRAME//F_IN_GOF and fov_count>player_fov_count:#
                if player_fractional_time+0.25>=VIDEO_GOF_LEN:
                    player_fractional_time=0
                    player_fov_count+=1
                    time_stamp+=VIDEO_GOF_LEN-player_fractional_time
                else:
                    time_stamp+=0.25
                    player_fractional_time+=0.25
                continue
            if back:
                continue
        if back==0 and fov_count<FRAME//F_IN_GOF:#new 
            selected_tile=fov_predict.predict(time,int(player_fov_count*F_IN_GOF+player_fractional_time*F_IN_GOF/VIDEO_GOF_LEN),fov_count*F_IN_GOF,(fov_count+1)*F_IN_GOF-1)
            selected_tile=[ math.ceil(sum(col) / len(col)) for col in zip(*selected_tile)]
            bit_rate=[0]*TILE_IN_F
            bw_constrain= net_env.predict_bw()
            bit_constrain=bw_constrain*VIDEO_GOF_LEN
            min_size=0
            for i in range(F_IN_GOF):
                for t in range(TILE_IN_F):
                    # if selected_tile[t]:
                    min_size+=video_size[fov_count*F_IN_GOF+i][t][0]
                    pass
            if min_size>=bit_constrain:
                pass
            else:
                cur_dis=dis[fov_count*F_IN_GOF]
                sorted_dis=sorted(cur_dis)
                index_dis = [cur_dis.index(x) for x in sorted_dis]
                for cur_l in range(QUALITY_LEVELS-1, 0, -1):
                    if min_size>=bit_constrain:
                        break
                    for tile in index_dis:
                        add_bit=0
                        sub_bit=0
                        for f in range(F_IN_GOF):
                            add_bit+=video_size[fov_count*F_IN_GOF+f][tile][cur_l]
                            sub_bit+=video_size[fov_count*F_IN_GOF+f][tile][0]
                        if selected_tile[tile] and  min_size+add_bit-sub_bit<=bit_constrain and bit_rate[tile]==0:
                            bit_rate[tile]=cur_l
                            min_size=min_size+add_bit-sub_bit            
            selected_tile=[1]*TILE_IN_F

            delay,buffer=net_env.get_video_gof_new(selected_tile,bit_rate)
            if player_fov_count*VIDEO_GOF_LEN+player_fractional_time+delay<fov_count*VIDEO_GOF_LEN:
                player_fov_count+=int((player_fractional_time+delay)//VIDEO_GOF_LEN)
                player_fractional_time=(player_fractional_time+delay)-int((player_fractional_time+delay)//VIDEO_GOF_LEN)*VIDEO_GOF_LEN
            else:
                sum_reward_rebuffer+=(player_fov_count*VIDEO_GOF_LEN+player_fractional_time+delay-fov_count*VIDEO_GOF_LEN)*REBUF_PENALTY
                player_fov_count=fov_count
                player_fractional_time=0
            time_stamp+=delay
            log_file.write(str(round(time_stamp,3)) + '\t' + str(player_fov_count+player_fractional_time/VIDEO_GOF_LEN)+ '\t' + 
                            'new' +str(fov_count)+ '\t'+
                            str(round(sum_reward_rebuffer,1)) + '\t'+ 
                            str(selected_tile) + '\t'+ 
                            str(bit_rate)+'\n')  
            fov_count+=1
            continue
        break       
    tp=tn=fp=fn=0 
    seen_tile=0
    last_bit_rate=[0]*TILE_IN_F
    for fov_count in range(int(FRAME/F_IN_GOF)):
        seen=[0]*TILE_IN_F
        for s in range(TILE_IN_F):
            sum_reward_switch+=SMOOTH_PENALTY * np.abs(MULTIPLE_QUALITY_LEVELS[max(buffer[fov_count][s],0)]-MULTIPLE_QUALITY_LEVELS[last_bit_rate[s]])
            last_bit_rate[s]=max(buffer[fov_count][s],0)
            for ff in range(F_IN_GOF):
                if fov[fov_count*F_IN_GOF+ff][s]:
                    seen[s]=1
                    continue
        for i in range(F_IN_GOF):
            for j in range(TILE_IN_F):
                seen_tile+=(buffer[fov_count][j]>=0)*seen[j]*video_size[fov_count*F_IN_GOF+i][j][buffer[fov_count][j]]/dis[fov_count*F_IN_GOF+i][j]
                if buffer[fov_count][j]==-1 and fov[fov_count*F_IN_GOF+i][j]==0:
                    tn+=1
                elif buffer[fov_count][j]>=0 and fov[fov_count*F_IN_GOF+i][j]==1:
                    tp+=1
                elif buffer[fov_count][j]==-1 and fov[fov_count*F_IN_GOF+i][j]==1:
                    fn+=1
                elif buffer[fov_count][j]>=0 and fov[fov_count*F_IN_GOF+i][j]==0:
                    fp+=1
    sum_reward_quality=INIT_QOE*seen_tile

    # log_file.write(str(round(time_stamp,8)) + '\t'+ '\t' + 
    #                 str(round(reward_quality,1)) + '\t'+ '\t'+
    #                 str(round(reward_rebuffer,1)) + '\t'+ '\t'+
    #                 str(round(reward_switch,1)) + '\t'+ '\t'+
    #                 str(round(reward,1)) +  '\t'+ '\t'+
    #                 str(round(buffer_size,1))+  '\t'+ '\t'+
    #                 str(bit_rate)+'\n')
    # log_file.flush()

    result_path = RESULT_FILE
    result_file = open(result_path, 'a')
    result_file.write(str((sum_reward_quality-sum_reward_rebuffer-sum_reward_switch)/len(video_size))+'\n')    

    open(result_path+'quali', 'a').write(str(sum_reward_quality/len(video_size))+'\n')
    open(result_path+'rebuf', 'a').write(str(sum_reward_rebuffer/len(video_size))+'\n') 
    open(result_path+'switch', 'a').write(str(sum_reward_switch/len(video_size))+'\n')
    open(result_path+'acc', 'a').write(str((tp+tn)/(tp+tn+fp+fn))+'\n')
    open(result_path+'recall', 'a').write(str((tp)/(tp+fn))+'\n') 
    open(result_path+'prec', 'a').write(str((tp)/(tp+fp))+'\n') 
    open(result_path+'buffer', 'a').write(str(buffer)+'\n') 
    return [sum_reward_quality/len(video_size),sum_reward_rebuffer/len(video_size),sum_reward_switch/len(video_size)]
