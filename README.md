# pointcloud_video 
This is a prototype designed for streaming point cloud data. It consists of two main components: a downloader and a player.  
An example video is point_1.gif  
## Prerequisites 
windows 10  
python 3.10  
open3d-python 0.17.0  
## How to use it  
Unzip tile data(releases) to /data/longdress/tiles  
You should run both the downloader and the player simultaneously.  
`python player.py`  
`python downloader.py`  
The downloader invokes a bitrate adaptation algorithm in real-time to determine the download target, while the player sequentially plays the video content from the buffer.  
Given the file size limitation, this is just an example program. You can modify the contents of the data folder to enable the playback of additional videos.  
## FoV model comparison  
The training, testing, and evaluation of four FoV prediction models were conducted within the "fov" folder. For each model, the specific steps included building the model architecture (such as neural networks), model training (using data from 30 users as the training set, while retaining the best-performing model file), and model testing (utilizing data from 10 users as the test set).The performance testing of the model encompasses both absolute error measurements and visibility tests for each individual CELL, which are crucial for video transmission QoE.  
You can freely replace the desired FoV prediction model in downloader.py  
## Algorithm comparison  
See /model_compare. 
## Evaluation results  
All results assessment and plotting for various models are located within the pointcloud_video folder.  
![image](https://github.com/shuqn/pointcloud_video/blob/main/point_1.gif)
