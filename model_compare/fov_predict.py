import numpy as np
import load_data
import Hyperparameters
import math
import pandas as pd
import csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import cook_data
import open3d as o3d
import os
# from sklearn.model_selection import train_test_split
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader
QUALITY_LEVELS =  Hyperparameters.QUALITY_LEVELS
REBUF_PENALTY = Hyperparameters.REBUF_PENALTY  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = Hyperparameters.SMOOTH_PENALTY
DEFAULT_QUALITY = Hyperparameters.DEFAULT_QUALITY  # default video quality without agent
RANDOM_SEED = Hyperparameters.RANDOM_SEED
RESEVOIR = 0.51  # BB
CUSHION = 2  # BB
current_directory = str(os.path.dirname(os.path.realpath(__file__)))
LOG_FILE = current_directory+'/results/log_sim_bb'
RESULT_FILE=current_directory+'/results/result_sim_bb'
DATA_PATH=current_directory+'/data'
COOKED_DATA_PATH=current_directory+'/cooked_data'
INIT_QOE=Hyperparameters.INIT_QOE
MULTIPLE_QUALITY_LEVELS=Hyperparameters.MULTIPLE_QUALITY_LEVELS
F_IN_GOF=Hyperparameters.F_IN_GOF
TILE_IN_F=Hyperparameters.TILE_IN_F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device= torch.device('cpu')
data = pd.read_csv(current_directory+'/data/Views/Video_1.csv')
used_user=30
num_users = 10
rows_per_user = 300
frames_for_prediction = 50
peidiction_window=90
class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.fc1 = nn.Linear(hidden_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            # print(len(x))
            # LSTM 1
            lstm1_out, _ = self.lstm1(x.view(len(x), 1, -1))
            # LSTM 2
            lstm2_out, _ = self.lstm2(lstm1_out.view(len(x), -1))
            # Fully Connected layers
            fc1_out = self.fc1(lstm2_out.view(len(x), -1))
            fc2_out = self.fc2(fc1_out.view(len(x), -1))
            
            return fc2_out
model=[]
dof_name=['x','y','z','rx','ry','rz']
for cur_model in range(len(dof_name)):
    input_size = frames_for_prediction
    hidden_size = 128  # LSTM隐藏层的大小
    output_size = 1 
    if cur_model>=3:
        input_size = frames_for_prediction*2
        output_size = 2  # 输入特征的大小（假设每帧有多少维数据）
    model.append(LSTMModel(input_size, hidden_size, output_size))
    model[cur_model].load_state_dict(torch.load(current_directory+'/fov_model/best_model_'+dof_name[cur_model]+'.pth', map_location=torch.device('cpu')))
    model[cur_model].eval()
    model[cur_model] = model[cur_model].to(device)
def predict(time,known,start,end):
    result_dof=[pd.DataFrame(columns=['user', 'frame', 'window','x']),pd.DataFrame(columns=['user', 'frame', 'window','y']),pd.DataFrame(columns=['user', 'frame', 'window','z']),pd.DataFrame(columns=['user', 'frame', 'window','rx']),pd.DataFrame(columns=['user', 'frame', 'window','ry']),pd.DataFrame(columns=['user', 'frame', 'window','rz'])]
    dof_name=['x','y','z','rx','ry','rz']
    # record_loss=[[],[],[],[],[],[]]
    for cur_model in range(len(dof_name)):

        # 输出的大小（假设预测的是多少维数据）
        # model =   # 重新创建模型实例


        with torch.no_grad():    
            for i in range(1):
                # user_start = (time-1) * rows_per_user
                # user_end = (time) * rows_per_user
                # user_fov_data = data.iloc[user_start:user_end, cur_model+1]  # 假设数据的六维特征从第二列开始
                
                if known>=49:
                    user_fov_data = data.iloc[(time-1) * rows_per_user+known-49:(time-1) * rows_per_user+known+1, cur_model+1]
                else:
                    user_fov_data=data.iloc[(time-1) * rows_per_user:(time-1) * rows_per_user+known+1, cur_model+1]
                    for less in range(49-known):
                        # print(data.iloc[(time-1) * rows_per_user])
                        user_fov_data=pd.concat([data.iloc[(time-1) * rows_per_user:(time-1) * rows_per_user+1, cur_model+1],user_fov_data], ignore_index=True)
                user_fov_array = user_fov_data.values.reshape((50, -1))  # 转换成 numpy 数组
                if cur_model>=3:
                    user_fov_array = [np.concatenate([value, value]) for value in user_fov_array]
                    for tr in range(len(user_fov_data)):
                    
                    # user_fov_array[tr]=np.append(user_fov_array[tr],0)
                    # print(user_fov_array[tr])
                    # print(np.array([np.sin(np.radians(user_fov_array[tr][0])),np.cos(np.radians(user_fov_array[tr][0]))]))
                        user_fov_array[tr]=np.array([np.sin(np.radians(user_fov_array[tr][0])),np.cos(np.radians(user_fov_array[tr][0]))])

                # 使用每 50 帧的数据预测下一帧的数据
                for j in range(1):
                    features = []
                    # targets = []
                    features.append(user_fov_array)
                    # targets.append(user_fov_array)
                    if j==170:
                        pass
                    for k in range(known+1,end+1):
                        
                        features_1 = torch.tensor(np.array(features), dtype=torch.float32)
                        # targets_1 = torch.tensor(targets, dtype=torch.float32)
                        features_1 = features_1.to(device)
                        # targets_1 = targets_1.to(device)
                        # print(features_1[0][0])
                        test_outputs = model[cur_model](features_1)
                        # print(test_outputs[0])
                        data_to_insert = {'user': time, 'frame': k, 'window': k-known,dof_name[cur_model]:test_outputs.tolist()[0][0]}
                        if cur_model>=3:
                            # print(test_outputs.tolist()[0])
                            # print(test_outputs.tolist()[0][0])
                            # print(test_outputs.tolist()[0][1])
                            # print(targets_1.tolist())
                            # print(targets_1.tolist()[0][0])
                            # print(targets_1.tolist()[0][1])
                            deg1=np.degrees(np.arctan2(test_outputs.tolist()[0][0], test_outputs.tolist()[0][1]))
                            # deg2=np.degrees(np.arctan2(targets_1.tolist()[0][0], targets_1.tolist()[0][1]))
                            if deg1<0:
                                deg1+=360
                            # if deg2<0:
                            #     deg2+=360
                            # if deg1-deg2>180:
                            #     deg2+=360
                            # if deg2-deg1>180:
                            #     deg1+=360
                            # test_loss=criterion(torch.tensor(deg1),torch.tensor(deg2))
                            data_to_insert = {'user': time, 'frame': k, 'window': k-known,dof_name[cur_model]:deg1} 
                        if k>=start:
                            result_dof[cur_model] = pd.concat([result_dof[cur_model], pd.DataFrame(data_to_insert, index=[0])], ignore_index=True)
                        # result_dof[cur_model] = result_dof[cur_model].append(data_to_insert, ignore_index=True)
                        # record_loss[cur_model][k].append(test_loss.cpu().item())
                        # features[0].append(np.array([test_outputs.item()]))
                        if cur_model>=3:
                            features[0].append(np.array(test_outputs.tolist()[0]))
                        else:
                            features[0]=np.append(features[0],np.array([test_outputs.item()]))
                        features[0]=features[0][1:]
                        # targets[0]=user_fov_array[j+frames_for_prediction+k]
    # print(result_dof)
    # df1 = pd.read_csv('E:/code/Volumetric-Video-Streaming/fov_model/prediction_x.csv')
    # df2 = pd.read_csv('E:/code/Volumetric-Video-Streaming/fov_model/prediction_y.csv')
    # df3 = pd.read_csv('E:/code/Volumetric-Video-Streaming/fov_model/prediction_z.csv')
    # df4 = pd.read_csv('E:/code/Volumetric-Video-Streaming/fov_model/prediction_rx.csv')
    # df5 = pd.read_csv('E:/code/Volumetric-Video-Streaming/fov_model/prediction_ry.csv')
    # df6 = pd.read_csv('E:/code/Volumetric-Video-Streaming/fov_model/prediction_rz.csv')

    result = result_dof[0].merge(result_dof[1], on=['user', 'frame', 'window'], how='inner')
    result = result.merge(result_dof[2], on=['user', 'frame', 'window'], how='inner')
    result = result.merge(result_dof[3], on=['user', 'frame', 'window'], how='inner')
    result = result.merge(result_dof[4], on=['user', 'frame', 'window'], how='inner')
    result = result.merge(result_dof[5], on=['user', 'frame', 'window'], how='inner')
    result = result.sort_values('frame', ascending=True)
    # print(result)
    ret=[]
    for i in range(len(result)):
        # print(DATA_PATH+'/Videos/Video_1/longdress_vox10_'+str(1051+result.iloc[i,2])+'.ply')
        file_path=DATA_PATH+'/Videos/Video_1/longdress_vox10_'+str(1051+result.iloc[i,2])+'.ply'
        ret.append(cook_data.make_fov_tiles(o3d.io.read_point_cloud(file_path),result.iloc[i,3:])[0])
    return ret
    # b