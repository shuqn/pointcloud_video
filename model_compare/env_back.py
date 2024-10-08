import numpy as np
import Hyperparameters
RANDOM_SEED = Hyperparameters.RANDOM_SEED
#每个gof的时间!!!!!!!!!!!!!!!!!!
VIDEO_GOF_LEN = Hyperparameters.VIDEO_GOF_LEN #秒
#每个GOF有N个'F'
F_IN_GOF=Hyperparameters.F_IN_GOF
#一个点云切块2*2*2
TILE_IN_F=Hyperparameters.TILE_IN_F
PACKET_PAYLOAD_PORTION =Hyperparameters.PACKET_PAYLOAD_PORTION
DECODING_TIME_RATIO=Hyperparameters.DECODING_TIME_RATIO
FRAME=Hyperparameters.FRAME
class Environment:
    def __init__(self, cooked_time, cooked_bw,video_size,random_seed=RANDOM_SEED):
        
        np.random.seed(random_seed)

        self.cooked_time = cooked_time
        self.cooked_bw = cooked_bw

        self.video_frame_counter = 0
        self.buffer_size = 0
        self.mahimahi_start_ptr = 1
        self.mahimahi_ptr = 1
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]
        self.video_size=video_size
        self.buffer=[]
        for i in range(FRAME//F_IN_GOF):
            self.buffer.append([])
            for j in range(TILE_IN_F):
                self.buffer[i].append(-1)        
    def get_video_gof_new(self, selected_tile,selected_quality):
        delay = 0.0  
        video_gof_counter_sent = 0  
        cur_gof_size=0
        for frame in range(F_IN_GOF):
            for tile in range(TILE_IN_F):
                if selected_tile[tile]>0.1:
                    cur_gof_size+=self.video_size[self.video_frame_counter+frame][tile][selected_quality[tile]]
        # print(cur_gof_size,tcnt)    
        delay+=cur_gof_size*DECODING_TIME_RATIO#decoding time
        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr]
            duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_gof_counter_sent + packet_payload > cur_gof_size:
                fractional_time=(cur_gof_size-video_gof_counter_sent)/throughput/PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                break

            video_gof_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0
                pass
        for tile in range(TILE_IN_F):
            if selected_tile[tile]>0.1:
                self.buffer[int(self.video_frame_counter/F_IN_GOF)][tile]=selected_quality[tile]
        self.video_frame_counter += F_IN_GOF
        end_of_video = False
        if self.video_frame_counter>= len(self.video_size)-1:
            end_of_video = True
        return delay,self.buffer
    def get_video_gof_back(self, selected_gof,selected_tile,selected_quality):
        #现在假设每个gof内的每个frame：quality不变，且不同的tile选用相同的quality

        delay = 0.0  
        video_gof_counter_sent = 0  
        cur_gof_size=0
        for frame in range(F_IN_GOF):
            for tile in range(TILE_IN_F):
                if selected_tile[tile]>0.1:
                    cur_gof_size+=self.video_size[selected_gof*F_IN_GOF+frame][tile][selected_quality[tile]]
        # print(cur_gof_size,tcnt)    
        delay+=cur_gof_size*DECODING_TIME_RATIO#decoding time
        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr]
            duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_gof_counter_sent + packet_payload > cur_gof_size:
                fractional_time=(cur_gof_size-video_gof_counter_sent)/throughput/PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                break

            video_gof_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0
                pass
        for j in range(TILE_IN_F):
            if selected_tile[j]==1:
                self.buffer[selected_gof][j]=selected_quality[j]
        return delay,self.buffer
    def predict_bw(self):
        return self.cooked_bw[self.mahimahi_ptr-1]*0.85+self.cooked_bw[self.mahimahi_ptr]*0.15