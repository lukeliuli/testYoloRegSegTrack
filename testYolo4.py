import cv2
import numpy as np
import torch
from ultralytics import YOLO
import sys
from pathlib import Path



from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet

class YOLODeepSORT:
    def __init__(self):
        # 初始化YOLO模型
        try:
            self.yolo = YOLO('yolo11n.pt')
        except Exception as e:
            print(f"YOLO模型加载失败: {e}")
            exit(1)

        # 初始化DeepSORT
        max_cosine_distance = 0.3
        nn_budget = None
        
        # 初始化深度特征提取器
        model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        
        # 创建跟踪器
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 
                                                         max_cosine_distance, 
                                                         nn_budget)
        self.tracker = Tracker(metric)
        
        # 目标类别
        self.target_classes = {
            0: 'person',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            1: 'bicycle'
        }
        
        # 类别对应的颜色
        self.colors = {
            'person': (0, 255, 255),     # 黄色
            'car': (0, 255, 0),          # 绿色
            'motorcycle': (255, 0, 0),    # 蓝色
            'bus': (0, 0, 255),          # 红色
            'truck': (255, 255, 0),      # 青色
            'bicycle': (255, 0, 255)     # 粉色
        }
        
        # 轨迹历史
        self.track_history = {}

    def detect_and_track(self, frame):
        """检测并跟踪目标"""
        # YOLO检测
        results = self.yolo(frame)
        detections = []
        
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf)
                cls = int(box.cls)
                
                if conf > 0.5 and cls in self.target_classes:
                    bbox = [x1, y1, x2-x1, y2-y1]  # [x,y,w,h]
                    detections.append(bbox)
        
        # 特征提取
        features = self.encoder(frame, detections)
        
        # 转换为Detection对象
        detection_list = []
        for bbox, feature in zip(detections, features):
            detection_list.append(Detection(bbox, 1.0, feature))
        
        # 更新跟踪器
        self.tracker.predict()
        self.tracker.update(detection_list)
        
        return self.tracker.tracks

    def draw_tracks(self, frame, tracks):
        """绘制跟踪结果"""
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            bbox = track.to_tlbr()  # 获取边界框 [top, left, bottom, right]
            track_id = track.track_id
            
            # 确保坐标为整数且在有效范围内
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # 计算中心点
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            # 更新轨迹历史
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(center)
            if len(self.track_history[track_id]) > 30:  # 保持最近30帧
                self.track_history[track_id].pop(0)
            
            # 绘制边界框
            color = (0, 255, 0)  # 使用绿色
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制ID（添加背景以提高可读性）
            label = f'ID: {track_id}'
            (label_w, label_h), baseline = cv2.getTextSize(label, 
                                                         cv2.FONT_HERSHEY_SIMPLEX, 
                                                         0.5, 2)
            cv2.rectangle(frame, (x1, y1-label_h-10), 
                         (x1+label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 绘制轨迹
            if len(self.track_history[track_id]) > 1:
                points = np.array(self.track_history[track_id], dtype=np.int32)
                cv2.polylines(frame, [points], False, color, 2)

    def process_video(self, video_path=0, output_path=None):
        """处理视频"""
        if isinstance(video_path, str):
            if not Path(video_path).exists():
                print(f"错误：找不到视频文件 {video_path}")
                return
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("错误：无法打开视频源")
            return
            
        window_name = 'YOLO+DeepSORT Tracking'
        cv2.namedWindow(window_name)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if output_path:
            output = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps, (width, height)
            )
            
        print("=== YOLO+DeepSORT 目标跟踪系统 ===")
        print("使用说明:")
        print("1. 按'q'键退出程序")
        print("2. 按's'键保存当前帧")
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 检测和跟踪
            tracks = self.detect_and_track(frame)
            
            # 绘制跟踪结果
            self.draw_tracks(frame, tracks)
            
            # 显示帧计数和目标数量
            cv2.putText(frame, f'Frame: {frame_count} Objects: {len(tracks)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 显示帧
            cv2.imshow(window_name, frame)
            
            if output_path:
                output.write(frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'frame_{frame_count}.jpg', frame)
                print(f"保存帧: frame_{frame_count}.jpg")
            
        cap.release()
        if output_path:
            output.release()
        cv2.destroyAllWindows()

def main():
    tracker = YOLODeepSORT()
    
    video_path = 'test_1.mp4'
    output_path = 'output_yolo_deepsort.mp4'
    
    tracker.process_video(video_path, output_path)

if __name__ == '__main__':
    main()
