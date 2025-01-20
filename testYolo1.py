
'''
给出基于opencv和yolo8的交通路面和交通参与者检测代码。要求:
1.检测和跟踪路面上的行人和各种车辆 。
2 输入是视频文件。

cursor 提供的代码如下:
'''

import cv2
import numpy as np
from ultralytics import YOLO
import time

def process_video(video_path, model_path):
    # 加载YOLOv8预训练模型
    model = YOLO(model_path)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频的基本信息
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 创建视频写入器
    output_path = 'output_detection.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    # 感兴趣的类别（基于COCO数据集）
    target_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # 开始计时
        start_time = time.time()
        
        # 运行YOLOv8检测
        results = model(frame)
        
        # 计算FPS
        fps = 1.0 / (time.time() - start_time)
        
        # 处理检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取类别
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                # 只处理目标类别
                if class_name in target_classes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 获取置信度
                    conf = float(box.conf[0])
                    
                    # 绘制边界框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 添加标签
                    label = f'{class_name}: {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 显示FPS
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 写入帧
        out.write(frame)
        
        # 显示结果
        cv2.imshow('Traffic Detection', frame)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 视频文件路径
    video_path = "test_1.mp4"
    video_path = "traffic.mp4"
    
    # YOLOv8模型路径（可以使用'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 或 'yolov8x.pt'）
    model_path = "yolov8n.pt"
    model_path = "yolo11n.pt"
    
    # 处理视频
    process_video(video_path, model_path)
