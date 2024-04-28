# pointcloud_video 
This is a prototype designed for streaming point cloud data. It consists of two main components: a downloader and a player.
## Prerequisites 
windows 10  
python 3.10  
open3d-python  
## How to use it  
You should run both the downloader and the player simultaneously.  
`pyhton player.py`  
`pyhton downloader.py`  
The downloader invokes a bitrate adaptation algorithm in real-time to determine the download target, while the player sequentially plays the video content from the buffer.  
Given the file size limitation, this is just an example program. You can modify the contents of the data folder to enable the playback of additional videos.
![image](https://github.com/shuqn/pointcloud_video/point_1.gif)
