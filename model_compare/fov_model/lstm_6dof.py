import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 加载 CSV 数据
data = pd.read_csv('E:/code/Volumetric-Video-Streaming/model3/data/Views/Video_1.csv')

# 假设每个用户的数据占据连续的 300 行，共 40 个用户
#30个用户用于训练
num_users = 30
rows_per_user = 300
frames_for_prediction = 50

# 整理数据集

dof_name=['x','y','z','rx','ry','rz']
for cur_dof in range(len(dof_name)):
    # 按用户构建数据集
    features = []
    targets = []
    for i in range(num_users):
        user_start = i * rows_per_user
        user_end = (i + 1) * rows_per_user
        user_fov_data = data.iloc[user_start:user_end, cur_dof+1]  # 假设数据的六维特征从第二列开始
        user_fov_array = user_fov_data.values.reshape((rows_per_user, -1))  # 转换成 numpy 数组
        if cur_dof>=3:
            user_fov_array = [np.concatenate([value, value]) for value in user_fov_array]
            for tr in range(rows_per_user):
                
                # user_fov_array[tr]=np.append(user_fov_array[tr],0)
                # print(user_fov_array[tr])
                # print(np.array([np.sin(np.radians(user_fov_array[tr][0])),np.cos(np.radians(user_fov_array[tr][0]))]))
                user_fov_array[tr]=np.array([np.sin(np.radians(user_fov_array[tr][0])),np.cos(np.radians(user_fov_array[tr][0]))])
        # 使用每 50 帧的数据预测下一帧的数据
        for j in range(0, rows_per_user - frames_for_prediction):
            features.append(user_fov_array[j:j+frames_for_prediction])
            targets.append(user_fov_array[j+frames_for_prediction])

    # 将列表转换为 PyTorch 张量
    features = torch.tensor(features, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    features = features.to(device)
    targets = targets.to(device)
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    # 定义 LSTM 模型
    # class LSTMModel(nn.Module):
    #     def __init__(self, input_size, hidden_size, output_size):
    #         super(LSTMModel, self).__init__()
    #         self.hidden_size = hidden_size
    #         self.lstm = nn.LSTM(input_size, hidden_size)
    #         self.fc = nn.Linear(hidden_size, output_size)
        
    #     def forward(self, x):
    #         # print(x.view(len(x), 1, -1))
    #         lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
    #         output = self.fc(lstm_out.view(len(x), -1))
    #         return output
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
    # 初始化模型和优化器
    print(features.shape)
    input_size = frames_for_prediction*features.shape[2]  # 输入特征的大小（假设每帧有多少维数据）
    hidden_size = 128  # LSTM隐藏层的大小
    output_size = targets.shape[1]  # 输出的大小（假设预测的是多少维数据）

    model = LSTMModel(input_size, hidden_size, output_size)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    writer = SummaryWriter(log_dir='E:/lstm')

    # 训练模型
    num_epochs = 1500
    min_loss=9999
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # 将 batch_size 设置为 8

    # 在训练过程中使用 train_loader
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            # 使用 batch_X 和 batch_y 进行模型训练
            model.train()
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            # 记录训练损失到 TensorBoard
            writer.add_scalar('Training Loss', loss.item(), epoch)
        with torch.no_grad():
            test_outputs = model(X_test)
            # print(test_outputs)
            test_loss = criterion(test_outputs, y_test)
            # print(f'Test Loss: {test_loss.item()}')
            if min_loss>test_loss.item():
                min_loss=test_loss.item()
                torch.save(model.state_dict(), 'E:/code/Volumetric-Video-Streaming/fov_model/best_model_'+dof_name[cur_dof]+'.pth')
        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    # 关闭 TensorBoard writer
    writer.close()

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        print(test_outputs)
        test_loss = criterion(test_outputs, y_test)
        print(f'Test Loss: {test_loss.item()}')


    model = LSTMModel(input_size, hidden_size, output_size)  # 重新创建模型实例
    model.load_state_dict(torch.load('E:/code/Volumetric-Video-Streaming/fov_model/best_model_'+dof_name[cur_dof]+'.pth'))
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        test_outputs = model(X_test)
        print(test_outputs)
        test_loss = criterion(test_outputs, y_test)
        print(f'Test Loss: {test_loss.item()}')