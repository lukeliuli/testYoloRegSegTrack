'''
一、运行yoloP2
1.修改一些地方
在demo.py中，修改
model  = torch.jit.load(weights） =>  model  = torch.jit.load(weights,map_location='cpu')
2.运行程序步骤
conda activate ./envYoloP2
cd YoloPv2-main
python demo.py  --device cpu --source ../traffic.mp4

3.结果分析
A.车辆分割效果好
B.道路分割效果也好
C.车道路线效果一般，不适合固定摄像，但是确实适合车载摄像头

二、运行yoloP

1.运行程序步骤
conda activate ./envYoloP2
cd YoloP-main
python tools/demo.py --source ../test_1.mp4
python tools/demo.py --source ../traffic1.jpg
2. 结果分析
A.车辆分割效果一般
B.道路分割效果一般
C.车道路线效果一般，不适合固定摄像，但是确实适合车载摄像头,
D.效果比YoloP2差很多

无论yolop还是yolop的结果，都显示输入数据决定了只适合车载前置摄像头，而不是固定路侧摄像头
'''