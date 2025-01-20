


# 给出基于opencv和yolov11的交通路面和交通参与者检测代码。
+ 要求:
1. 检测和跟踪路面上的行人和各种车辆. 
2. 输入是视频文件
---
## testYolo1.py 给出基于opencv和yolo11和yolov8的交通参与者检测代码
---
## testYolo2.py 给出基于opencv和yolo11和yolov8的带分割功能的和交通参与者检测代码
---
## testYolo3.py 给出基于opencv的yolo11和yolov8的车辆识别和跟踪代码,核心是跟踪,用yolo自带的。
---
## （放弃）testYolo4.py 给出基于opencv的yolo和yolovDeepsort,deepsort的车辆识别和跟踪代码,核心是跟踪。
+ 注意：
1. 激活test4YoloDeepsort环境 
2. 放弃原因是安装困难，而且似乎效果与yolo比较一般
---
##  testYolo5.py 给出基于yolo_sam的路面分割代码。
---
##  (失败)testYolo6.py  测试yolo-obb代码。
+ 始终无法获得任何结果，不知道为什么: the boxes in the result obtained are None
---
##  testYolo7.py 给出基于yolop和yolop2的yolo的路面分割代码。注意：请看文件内的说明

## 代码中还需要大量文件和文件夹，相关东西因为体积大，放在百度网盘中:
0. 百度网盘：myCodes/testYoloRegSegTrack(https://pan.baidu.com/s/1tH2JEJZyUWyWi4YEPL00ww?pwd=pbx5 提取码: pbx5):
1. envYoloP2: 运行yoloP和yoloP2的conda环境，python=3.10 (不用也OK，自己安装各种库)
2. inputs:各种测试用的图片和视频
3. models:各种yolo的预训练模型
4. YOLOP-main: YOLOP的代码
5. YOLOPv2-main: YOLOPv2的代码



