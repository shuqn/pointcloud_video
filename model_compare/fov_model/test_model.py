import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = pd.read_csv('E:/code/Volumetric-Video-Streaming/model3/data/Views/Video_1.csv')

# 假设每个用户的数据占据连续的 300 行，共 40 个用户
#后10个用户用于测试
used_user=30
num_users = 10
rows_per_user = 300
frames_for_prediction = 50
peidiction_window=90

result_dof=[pd.DataFrame(columns=['user', 'frame', 'window','x']),pd.DataFrame(columns=['user', 'frame', 'window','y']),pd.DataFrame(columns=['user', 'frame', 'window','z']),pd.DataFrame(columns=['user', 'frame', 'window','rx']),pd.DataFrame(columns=['user', 'frame', 'window','ry']),pd.DataFrame(columns=['user', 'frame', 'window','rz'])]
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


dof_name=['x','y','z','rx','ry','rz']
record_loss=[[],[],[],[],[],[]]
for cur_model in range(len(dof_name)):
    input_size = frames_for_prediction
    hidden_size = 128  # LSTM隐藏层的大小
    output_size = 1 
    if cur_model>=3:
        input_size = frames_for_prediction*2
        output_size = 2  # 输入特征的大小（假设每帧有多少维数据）
     # 输出的大小（假设预测的是多少维数据）
    model = LSTMModel(input_size, hidden_size, output_size)  # 重新创建模型实例
    model.load_state_dict(torch.load('E:/code/Volumetric-Video-Streaming/fov_model/best_model_'+dof_name[cur_model]+'.pth'))
    model.eval()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 整理数据集

    
    for i in range(peidiction_window):
        record_loss[cur_model].append([])
    with torch.no_grad():    
        for i in range(used_user,used_user+num_users):
            user_start = i * rows_per_user
            user_end = (i + 1) * rows_per_user
            user_fov_data = data.iloc[user_start:user_end, cur_model+1]  # 假设数据的六维特征从第二列开始
            user_fov_array = user_fov_data.values.reshape((rows_per_user, -1))  # 转换成 numpy 数组
            if cur_model>=3:
                user_fov_array = [np.concatenate([value, value]) for value in user_fov_array]
                for tr in range(rows_per_user):
                
                # user_fov_array[tr]=np.append(user_fov_array[tr],0)
                # print(user_fov_array[tr])
                # print(np.array([np.sin(np.radians(user_fov_array[tr][0])),np.cos(np.radians(user_fov_array[tr][0]))]))
                    user_fov_array[tr]=np.array([np.sin(np.radians(user_fov_array[tr][0])),np.cos(np.radians(user_fov_array[tr][0]))])

            # 使用每 50 帧的数据预测下一帧的数据
            for j in range(0, rows_per_user - frames_for_prediction-peidiction_window):
                features = []
                targets = []
                features.append(user_fov_array[j:j+frames_for_prediction])
                targets.append(user_fov_array[j+frames_for_prediction])
                if j==170:
                    pass
                for k in range(peidiction_window):
                    
                    features_1 = torch.tensor(features, dtype=torch.float32)
                    targets_1 = torch.tensor(targets, dtype=torch.float32)
                    features_1 = features_1.to(device)
                    targets_1 = targets_1.to(device)
                    test_outputs = model(features_1)
                    test_loss = criterion(test_outputs, targets_1)
                    data_to_insert = {'user': i+1, 'frame': j+frames_for_prediction+1+k, 'window': k,dof_name[cur_model]:test_outputs.tolist()[0][0]}
                    if cur_model>=3:
                        # print(test_outputs.tolist()[0])
                        # print(test_outputs.tolist()[0][0])
                        # print(test_outputs.tolist()[0][1])
                        # print(targets_1.tolist())
                        # print(targets_1.tolist()[0][0])
                        # print(targets_1.tolist()[0][1])
                        deg1=np.degrees(np.arctan2(test_outputs.tolist()[0][0], test_outputs.tolist()[0][1]))
                        deg2=np.degrees(np.arctan2(targets_1.tolist()[0][0], targets_1.tolist()[0][1]))
                        if deg1<0:
                            deg1+=360
                        if deg2<0:
                            deg2+=360
                        if deg1-deg2>180:
                            deg2+=360
                        if deg2-deg1>180:
                            deg1+=360
                        test_loss=criterion(torch.tensor(deg1),torch.tensor(deg2))
                        data_to_insert = {'user': i+1, 'frame': j+frames_for_prediction+1+k, 'window': k,dof_name[cur_model]:deg1} 
                    result_dof[cur_model] = pd.concat([result_dof[cur_model], pd.DataFrame(data_to_insert, index=[0])], ignore_index=True)
                    # result_dof[cur_model] = result_dof[cur_model].append(data_to_insert, ignore_index=True)
                    record_loss[cur_model][k].append(test_loss.cpu().item())
                    # features[0].append(np.array([test_outputs.item()]))
                    if cur_model>=3:
                        features[0].append(np.array(test_outputs.tolist()[0]))
                    else:
                        features[0]=np.append(features[0],np.array([test_outputs.item()]))
                    features[0]=features[0][1:]
                    targets[0]=user_fov_array[j+frames_for_prediction+k]
                    # print(test_loss.item())
for cur_model in range(len(dof_name)):
    # for cur_window in range(peidiction_window):
    #     with open('e:/code/Volumetric-Video-Streaming/fov_model/output'+str(cur_window)+'.txt', 'w') as file:
    #         # 将列表中的每个元素写入文件，每个元素占据一行
    #         for item in record_loss[4][cur_window]:
    #             file.write(str(item) + '\n')
    mean_values = [sum(sublist) / len(sublist) for sublist in record_loss[cur_model]]
    for i in range(peidiction_window):
        mean_values[i] = mean_values[i]**0.5
    x = list(range(peidiction_window))

    # 绘制折线图
    plt.plot(x, mean_values, marker='o')  # 使用 marker='o' 添加圆点标记
    plt.xlabel('prediction_length')
    plt.ylabel('RMSE')
    plt.title('fov预测RMSE随距离的变化')
    plt.show()
    print(sum(mean_values)/peidiction_window)
    result_dof[cur_model].to_csv('E:/code/Volumetric-Video-Streaming/fov_model/prediction_'+dof_name[cur_model]+'.csv', index=False)