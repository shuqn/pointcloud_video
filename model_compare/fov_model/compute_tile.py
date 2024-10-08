import pandas as pd

# 读取六个CSV文件
df1 = pd.read_csv('E:/code/Volumetric-Video-Streaming/fov_model/prediction_x.csv')
df2 = pd.read_csv('E:/code/Volumetric-Video-Streaming/fov_model/prediction_y.csv')
df3 = pd.read_csv('E:/code/Volumetric-Video-Streaming/fov_model/prediction_z.csv')
df4 = pd.read_csv('E:/code/Volumetric-Video-Streaming/fov_model/prediction_rx.csv')
df5 = pd.read_csv('E:/code/Volumetric-Video-Streaming/fov_model/prediction_ry.csv')
df6 = pd.read_csv('E:/code/Volumetric-Video-Streaming/fov_model/prediction_rz.csv')

result = df1.merge(df2, on=['user', 'frame', 'window'], how='inner')
result = result.merge(df3, on=['user', 'frame', 'window'], how='inner')
result = result.merge(df4, on=['user', 'frame', 'window'], how='inner')
result = result.merge(df5, on=['user', 'frame', 'window'], how='inner')
result = result.merge(df6, on=['user', 'frame', 'window'], how='inner')
# 使用pd.concat连接所有DataFrame，按行连接
# result = pd.concat([df1, df2, df3, df4, df5, df6], axis=1)

# 保存连接后的DataFrame为CSV文件
result.to_csv('E:/code/Volumetric-Video-Streaming/fov_model/prediction.csv', index=False)
