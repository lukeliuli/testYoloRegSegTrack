from ultralytics import YOLO
import cv2
import numpy as np
import torch

class SegmentationDemo:
    def __init__(self):
        # 加载预训练的分割模型
        # 可以选择不同大小的模型：yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt, yolov8l-seg.pt, yolov8x-seg.pt
        self.model = YOLO('yolov8n-seg.pt')
        
        # 定义颜色映射（为不同类别设置不同颜色）
        self.colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)
    
    def process_image(self, image_path):
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("无法读取图像")
            
        # 运行推理
        results = self.model(image)
        
        # 获取原始图像的副本
        annotated_image = image.copy()
        
        # 处理每个检测结果
        for result in results:
            # 获取分割掩码
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.data.cpu().numpy()
                cls = result.boxes.cls.cpu().numpy()
                
                # 处理每个实例
                for mask, box, class_id in zip(masks, boxes, cls):
                    # 创建彩色掩码
                    color = self.colors[int(class_id)]
                    colored_mask = np.zeros_like(image)
                    colored_mask[mask[0] > 0] = color
                    
                    # 将掩码叠加到图像上
                    alpha = 0.5  # 透明度
                    annotated_image = cv2.addWeighted(
                        annotated_image, 1,
                        colored_mask, alpha,
                        0
                    )
                    
                    # 绘制边界框
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color.tolist(), 2)
                    
                    # 添加类别标签
                    label = f'{self.model.names[int(class_id)]}: {box[4]:.2f}'
                    cv2.putText(
                        annotated_image, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2
                    )
        
        return annotated_image

    def process_video(self, video_path):
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")
            
   
        
        
        
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
                
            # 处理帧
            output_size = (640,384)
            frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR)

            results = self.model(frame)
            annotated_frame = frame.copy()
            
            # 处理分割结果
            for result in results:
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    boxes = result.boxes.data.cpu().numpy()
                    cls = result.boxes.cls.cpu().numpy()
                    print('cls:',cls)
                    for mask, box, class_id in zip(masks, boxes, cls):
                        
                        '''
                        color = self.colors[int(class_id)]
                        colored_mask = np.zeros_like(frame)
                        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        
                        annotated_frame = cv2.addWeighted(
                            annotated_frame, 0.5,
                            mask_rgb, 0.5,
                            0,dtype=cv2.CV_32F
                        )
                        '''

                        color = self.colors[int(class_id)]
                        colored_mask = np.zeros_like(frame)
                        index = mask > 0
                        colored_mask[index] = color
                        annotated_frame[index] = (annotated_frame[index]*0.1+colored_mask[index]*0.9).astype(np.uint8)
                       


                        # 绘制边界框和标签
                        x1, y1, x2, y2 = map(int, box[:4])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color.tolist(), 2)
                        label = f'{self.model.names[int(class_id)]}: {box[4]:.2f}'
                        cv2.putText(
                            annotated_frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2
                        )
            
            
            
            # 显示实时结果
            cv2.imshow('Segmentation', annotated_frame)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break
        
        # 释放资源
        cap.release()
    
        cv2.destroyAllWindows()

def main():
    # 创建分割器实例
    segmenter = SegmentationDemo()
    
    # 处理图像示例
    # image_path = 'path/to/your/image.jpg'
    # result_image = segmenter.process_image(image_path)
    # cv2.imwrite('segmentation_result.jpg', result_image)
    
    # 处理视频示例
    video_path = 'test_1.mp4'  # 替换为你的视频路径
    video_path = 'traffic.mp4'  # 替换为你的视频路径
    
    segmenter.process_video(video_path)

if __name__ == '__main__':
    main()