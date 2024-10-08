import random
import Hyperparameters
RANDOM_SEED = Hyperparameters.RANDOM_SEED
TILE_IN_F=Hyperparameters.TILE_IN_F  #一个点云切块2*3*2
QUALITY_LEVELS =Hyperparameters.QUALITY_LEVELS
MULTIPLE_QUALITY_LEVELS=Hyperparameters.MULTIPLE_QUALITY_LEVELS
random.seed(RANDOM_SEED)

def random_trace(bw_count=30000,init_bw=4):#这里的单位是 Mbps
    random_time=0.1
    random_bw=0.2
    cooked_time = [0]
    cooked_bw = [init_bw]
    for i in range(1,bw_count):
        cooked_time.append(i*0.4+random.uniform(-random_time, random_time)) 
        cooked_bw.append(max(min(cooked_bw[i-1]+random.uniform(-random_bw, random_bw),0),8)*20)
    return cooked_time, cooked_bw

def random_video(frame_count=5000,min_base_random_size=0.2,max_base_random_size=0.30): #单位是Mb
    video_size=[]
    for i in range(frame_count):
        video_size.append([])
        for j in range(TILE_IN_F):
            video_size[i].append([])
            base_size=random.uniform(min_base_random_size, max_base_random_size)
            for k in MULTIPLE_QUALITY_LEVELS:
                video_size[i][j].append(base_size*k)
    return video_size

def random_fov(fov_count=5000,choices = [0, 1],weights = [0.3, 0.7]):
    fov=[]
    for i in range(fov_count):
        fov.append([])
        for j in range(TILE_IN_F):
            fov[i].append(random.choices(choices, weights, k=1)[0])
    return fov


def load_trace(filename=None,bw_count=30000,init_bw=4,startposition=0):
    ####一定看好单位！！！
    data_list = []
    with open(filename, 'r') as file:
        # 跳过起始行之前的行
        for _ in range(startposition):
            next(file)
        i=0
        time_list=[]
        # 从起始行开始读取每一行的浮点数并添加到列表中
        for line in file:
            values =line.strip().split()
            time_list.append(float(values[1]))
            data_list.append(float(values[0])/1000)
            i+=1
        # print(sum(data_list)/len(data_list))
    return time_list,data_list
#每个点占24字节
def load_video(filename=None,frame_count=5000,min_base_random_size=0.2,max_base_random_size=0.30):
    video_size = []
    with open(filename, 'r') as file:
        for line in file:
            row = list(map(float, line.split()))
            row=[x*0.004*8/1000/8/MULTIPLE_QUALITY_LEVELS[-1] for x in row]#转为Mb 转为最小质量的占用
            video_size.append(row)
    ret=[]
    for i in range(len(video_size)):
        ret.append([])
        for j in range(len(video_size[i])):
            ret[i].append([])
            for k in MULTIPLE_QUALITY_LEVELS:
                ret[i][j].append(k*video_size[i][j])
    return ret
    # return random_video(frame_count,min_base_random_size,max_base_random_size)

def load_fov(filename=None,fov_count=5000,choices = [0, 1],weights = [0.3, 0.7]):
    fov = []
    with open(filename, 'r') as file:
        for line in file:
            row = list(map(float, line.split()))
            fov.append(row)
    return fov
def load_dis(filename=None,fov_count=5000,choices = [0, 1],weights = [0.3, 0.7]):
    dis = []
    with open(filename, 'r') as file:
        for line in file:
            row = list(map(float, line.split()))
            dis.append(row)
    return dis

    # return random_fov(fov_count,choices,weights)