# 点云传输仿真  
注意：ply数据放到/data/Video，需要自行下载 https://plenodb.jpeg.org/pc/8ilabs/#:~:text=The%20dynamic%20voxelized%20point%20cloud%20sequences%20in%20this%20dataset%20are
## 代码实现了以下功能  
(1)使用已有数据集 视频/网络/FOV 预处理+格式转换  
(2)PACE和其他几个baseline。包括bufferbased/下载全部tile/下载可见tile/ddqn强化学习法
(3)收集测试结果
## 一些解释  
test_xxx.py:仿真主程序入口。读取数据集并进行测试
load_data.py:读取(生成测试数据)  
env_back.py:仿真环境，模拟了传输流程  
Hyper.py:设置超参数和常量，防止不一致  
cook_data.py:原始数据预处理，转化格式。
$$
QoE_{frame}=A·QoE_{quality}-B·QoE_{rebuffer}-C·QoE_{switch}\\
 \\

QoE_{quality}=quality\_level·\sum_{frame}\sum_{tile}fov_{predicted}[frame][tile]·fov_{real}[frame][tile]\\
QoE_{rebuffer}=rebuffer\_time\\
QoE_{switch}=|quality\_level_{last\_frame}-quality\_level_{current\_frame}|
$$  

23.11.24 lsq