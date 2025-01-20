from ultralytics import FastSAM
import cv2
import numpy as np
import torch
import random

class FastSAMSegmenter:
    def __init__(self, model_path="FastSAM-s.pt"):
        """
        初始化FastSAM分割器
        Args:
            model_path: FastSAM模型路径
        """
        # 初始化模型
        self.model = FastSAM(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 分割参数
        self.conf_threshold = 0.4
        self.iou_threshold = 0.3
        
        # 可视化参数
        self.alpha = 0.5  # 遮罩透明度
        
    def generate_random_colors(self, num_colors):
        """
        生成随机颜色
        """
        colors = []
        for _ in range(num_colors):
            color = (random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255))
            colors.append(color)
        return colors

    def segment_image(self, image_path):
        """
        对图像进行分割
        Args:
            image_path: 输入图像路径
        """
        # 运行模型预测
        results = self.model(image_path,
                           device=self.device,
                           retina_masks=True,
                           conf=self.conf_threshold,
                           iou=self.iou_threshold)
        
        return results[0]

    def visualize_segments(self, image_path, result, output_path=None):
        """
        可视化分割结果
        """
        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("无法读取图像")
            
        # 获取图像尺寸
        height, width = image.shape[:2]
        
        # 创建叠加图像
        overlay = image.copy()
        
        # 获取分割掩码
        if result.masks is not None:
            masks = result.masks.data
            
            # 生成随机颜色
            colors = self.generate_random_colors(len(masks))
            
            # 处理每个掩码
            for idx, mask in enumerate(masks):
                # 将掩码转换为numpy数组
                mask_array = mask.cpu().numpy()
                
                # 调整掩码尺寸以匹配图像
                mask_array = cv2.resize(mask_array, (width, height))
                
                # 创建彩色掩码
                color_mask = np.zeros_like(image)
                color_mask[mask_array > 0.5] = colors[idx]
                
                # 叠加掩码
                overlay = cv2.addWeighted(overlay, 1, color_mask, self.alpha, 0)
                
                # 绘制轮廓
                binary_mask = (mask_array > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask, 
                                             cv2.RETR_EXTERNAL, 
                                             cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, colors[idx], 2)
        
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, overlay)
            print(f"结果已保存至: {output_path}")
        
        return overlay

    def process_image(self, image_path, output_path=None, show_result=True):
        """
        处理单张图片
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径
            show_result: 是否显示结果
        """
        try:
            # 运行分割
            print("正在进行分割...")
            result = self.segment_image(image_path)
            
            # 可视化结果
            print("正在生成可视化结果...")
            segmented_image = self.visualize_segments(image_path, result, output_path)
            
            # 显示结果
            if show_result:
                cv2.imshow('Segmentation Result', segmented_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return segmented_image
            
        except Exception as e:
            print(f"处理过程中出现错误: {str(e)}")
            return None

    def process_batch(self, input_folder, output_folder):
        """
        批量处理图像
        Args:
            input_folder: 输入文件夹路径
            output_folder: 输出文件夹路径
        """
        import os
        from pathlib import Path
        
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取所有图像文件
        image_files = [f for f in Path(input_folder).glob('*') 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        for image_file in image_files:
            print(f"\n处理图像: {image_file.name}")
            output_path = os.path.join(output_folder, f"seg_{image_file.name}")
            self.process_image(str(image_file), output_path, show_result=False)

def main():
    # 创建分割器实例
    segmenter = FastSAMSegmenter()
    
    # 处理单张图片
    image_path = "traffic1.jpg"  # 替换为你的图片路径
    output_path = "output_segmentation.jpg"
    
    # 方法1：处理单张图片并显示结果
    result = segmenter.process_image(image_path, output_path)
    
    # 方法2：批量处理文件夹中的图片
    # input_folder = "input_images"
    # output_folder = "output_images"
    # segmenter.process_batch(input_folder, output_folder)

if __name__ == "__main__":
    main()