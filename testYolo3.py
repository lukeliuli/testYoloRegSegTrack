import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import torch
from pathlib import Path

class VehicleTracker:
    def __init__(self):
        # 初始化YOLO模型
        try:
            self.model = YOLO('yolov8n.pt')
            self.model = YOLO('yolo11n.pt')
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("请确保已安装ultralytics并下载了yolov8x.pt模型")
            exit(1)
        
        # 目标类别ID和对应的中文名称
        self.target_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            1: 'bicycle',
            0: 'pedestrian'
        }
        
        # 类别对应的颜色映射
        self.colors = {
            'car': (0, 255, 0),    # 绿色
            'motorcycle': (255, 0, 0),    # 蓝色
            'bus': (0, 0, 255),    # 红色
            'truck': (255, 255, 0),    # 青色
            'bicycle': (255, 0, 255),  # 粉色
            'pedestrian': (0, 255, 255)     # 黄色
        }
        
        # 用于跟踪的数据结构
        self.track_history = defaultdict(lambda: [])
        
        # 统计数据
        self.vehicle_count = defaultdict(int)
        self.current_tracked = set()

    def draw_tracks(self, frame, tracks, line_thickness=2):
        """绘制跟踪轨迹"""
        # 清除之前帧的计数
        self.vehicle_count.clear()
        
        # 获取跟踪结果
        boxes = tracks.boxes
        for box in boxes:
            # 检查是否有跟踪ID
            if not hasattr(box, 'id'):
                continue
                
            track_id = int(box.id.item())
            cls = int(box.cls.item())
            
            if cls not in self.target_classes:
                continue
                
            # 获取边界框和置信度
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf)
            
            # 计算中心点
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            # 更新轨迹历史
            self.track_history[track_id].append(center)
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)
            
            # 获取目标类别的颜色
            class_name = self.target_classes[cls]
            color = self.colors[class_name]
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)
            
            # 添加标签（包含置信度）
            label = f'{class_name} #{track_id} {conf:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_thickness)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), line_thickness)
            
            # 绘制轨迹
            points = np.array(self.track_history[track_id], dtype=np.int32)
            cv2.polylines(frame, [points], False, color, line_thickness)
            
            # 更新统计信息
            self.current_tracked.add(track_id)
            self.vehicle_count[class_name] += 1

    def draw_stats(self, frame):
        """在画面上显示统计信息"""
        y_offset = 30
        for cls_name, count in self.vehicle_count.items():
            text = f'{cls_name}: {count}'
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            y_offset += 30

    def process_video(self, video_path, output_path=None):
        """处理视频文件"""
        if not Path(video_path).exists():
            print(f"错误：找不到视频文件 {video_path}")
            return
            
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            return
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 设置输出视频
        if output_path:
            output = cv2.VideoWriter(output_path, 
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 使用YOLO进行检测和跟踪
            results = self.model.track(frame, persist=True, classes=list(self.target_classes.keys()),tracker="bytetrack.yaml")
            
            if results and len(results) > 0:
                self.draw_tracks(frame, results[0])
            
            # 绘制统计信息
            self.draw_stats(frame)
            
            # 显示处理后的帧
            cv2.imshow('Vehicle Tracking', frame)
            
            if output_path:
                output.write(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if output_path:
            output.release()
        cv2.destroyAllWindows()

def main():
    # 创建跟踪器实例
    tracker = VehicleTracker()
    
    # 设置视频路径
    video_path = 'test_1.mp4'
    video_path = 'traffic.mp4'
    output_path = 'output_tracked.mp4'
    
    print("=== 车辆检测与跟踪系统 ===")
    print(f"处理视频: {video_path}")
    print("支持检测的目标类别:", ", ".join(tracker.target_classes.values()))
    print("按'q'键退出程序")
    
    tracker.process_video(video_path, output_path)

if __name__ == '__main__':
    main()
