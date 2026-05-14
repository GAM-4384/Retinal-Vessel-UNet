import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import random
from model import Unet
from dataloader import DataLoader

def load_trained_model(weights_path):
    """加载训练好的模型"""
    model = Unet(
        data_format='channels_last',
        classes=1,
        transpose_conv=False,
        name='unet_segmentation'
    )

    # 加载权重
    model.load_weights(weights_path)
    return model


def visualize_results(images, predictions, masks, save_path=None):
    """可视化结果"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    plt.subplots_adjust(hspace=0.3)

    titles = ['Original Image', 'Model Prediction', 'Ground Truth']

    for i in range(3):  # 3张图片
        # 原始图像
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f'Sample {i + 1}: {titles[0]}')
        axes[i, 0].axis('off')

        # 模型预测
        axes[i, 1].imshow(predictions[i], cmap='gray')
        axes[i, 1].set_title(f'Sample {i + 1}: {titles[1]}')
        axes[i, 1].axis('off')

        # Ground Truth
        axes[i, 2].imshow(masks[i], cmap='gray')
        axes[i, 2].set_title(f'Sample {i + 1}: {titles[2]}')
        axes[i, 2].axis('off')

    plt.suptitle('Model Evaluation Results', fontsize=16)

    if save_path:
        plt.savefig(save_path)
        print(f"结果已保存到: {save_path}")

    plt.show()


def evaluate_model(model_weights_path, num_samples=3):
    """评估模型"""
    try:
        # 加载数据
        data_loader = DataLoader(batch_size=1, image_size=(256, 256))
        test_dataset = data_loader.get_test_dataset()

        # 加载模型
        print("加载模型权重...")
        model = load_trained_model(model_weights_path)

        # 随机选择样本
        all_samples = list(test_dataset.as_numpy_iterator())
        selected_samples = random.sample(all_samples, num_samples)

        images = []
        predictions = []
        masks = []

        print("处理选中的样本...")
        for sample in selected_samples:
            image, mask = sample

            # 获取预测结果
            pred = model.predict(image, verbose=0)

            # 处理数据用于显示
            image_display = image[0]  # 移除批次维度
            pred_display = pred[0, ..., 0]  # 移除批次维度和通道维度
            mask_display = mask[0, ..., 0]  # 移除批次维度和通道维度

            images.append(image_display)
            predictions.append(pred_display)
            masks.append(mask_display)

        # 可视化结果
        print("生成可视化结果...")
        visualize_results(
            images,
            predictions,
            masks,
            save_path="evaluation_results.png"
        )

        print("评估完成！")

    except Exception as e:
        print(f"评估过程中发生错误: {str(e)}")
        raise e


if __name__ == '__main__':
    # 设置模型权重路径
    model_weights_path =  "E:/My struggle/U-Net/logs/final_model"

    print("开始模型评估...")
    evaluate_model(model_weights_path)