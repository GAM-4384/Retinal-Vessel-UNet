import os
import cv2
import numpy as np
from glob import glob
from pathlib import Path
from PIL import Image
import numpy as np
# 定义数据集路径
BASE_DIR = r"E:\My struggle\U-Net"
RAW_DIR = os.path.join(BASE_DIR, "raw")
ARCHIVE_DIR = os.path.join(RAW_DIR, "archive")
DRIVE_DIR = os.path.join(RAW_DIR, "DRIVE")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "preprocessed")


def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_image(image_path, target_size=(584, 565)):
    """加载图像并调整大小"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image


def load_mask(mask_path, target_size=(584, 565)):
    """加载mask并调整大小"""
    try:
        # 使用PIL读取图像
        with Image.open(mask_path) as img:
            # 如果是GIF，取第一帧
            if 'n_frames' in img.info:
                img.seek(0)
            # 转换为灰度图
            img = img.convert('L')
            # 调整大小
            img = img.resize(target_size)
            # 转换为numpy数组
            mask = np.array(img)
            # 二值化
            mask = (mask > 128).astype(np.uint8) * 255
            return mask
    except Exception as e:
        raise ValueError(f"无法读取mask: {mask_path}, 错误: {str(e)}")


def preprocess_image(image):
    """图像预处理"""
    # 1. 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # 2. 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray.astype(np.uint8))

    # 3. Gamma校正
    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype(np.uint8)
    gray = cv2.LUT(gray, table)

    # 4. 标准化到0-1范围
    gray = gray.astype(np.float32) / 255.0
    return gray


def safe_imwrite(path, img):
    """安全的图像保存函数"""
    try:
        # 确保目标目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 使用imencode和文件写入来代替imwrite
        success, encoded_img = cv2.imencode(os.path.splitext(path)[1], img)
        if success:
            with open(path, 'wb') as f:
                encoded_img.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(f"保存图像时出错 {path}: {str(e)}")
        return False


def process_archive_dataset():
    """处理archive数据集"""
    print("\n开始处理archive数据集...")

    # 获取所有的原始图像路径，并去重
    image_patterns = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
    image_paths = set()  # 使用集合来避免重复
    for pattern in image_patterns:
        paths = glob(os.path.join(ARCHIVE_DIR, "Image*" + pattern))
        image_paths.update(paths)

    # 转换为排序后的列表
    image_paths = sorted(list(image_paths))

    if not image_paths:
        print(f"警告: 在{ARCHIVE_DIR}中未找到图像文件")
        return

    print(f"找到{len(image_paths)}张原始图像")

    # 处理每张图像
    for idx, image_path in enumerate(image_paths):
        try:
            # 构建对应的mask文件名
            base_name = Path(image_path).stem  # 例如：Image_01L
            mask_path = os.path.join(ARCHIVE_DIR, f"{base_name}_1stHO.png")

            if not os.path.exists(mask_path):
                print(f"跳过{image_path}: 未找到对应的mask文件")
                continue

            # 加载并预处理图像
            image = load_image(image_path)
            processed_image = preprocess_image(image)
            processed_save = (processed_image * 255).astype(np.uint8)

            # 加载mask
            mask = load_mask(mask_path)

            # 确定保存目录（前40张为训练集，其余为测试集）
            split = 'training' if idx < 40 else 'test'
            split_idx = idx if idx < 40 else (idx - 40)

            # 保存路径
            image_save_dir = os.path.join(PREPROCESSED_DIR, split, 'images_pre')
            mask_save_dir = os.path.join(PREPROCESSED_DIR, split, 'mask')
            ensure_dir(image_save_dir)
            ensure_dir(mask_save_dir)

            # 保存文件
            image_save_path = os.path.join(image_save_dir, f"archive_{split_idx:02d}.png")
            mask_save_path = os.path.join(mask_save_dir, f"archive_{split_idx:02d}.png")

            if not safe_imwrite(image_save_path, processed_save):
                print(f"保存处理后的图像失败: {image_save_path}")
                continue

            if not safe_imwrite(mask_save_path, mask):
                print(f"保存mask失败: {mask_save_path}")
                continue

            print(f"已处理: {Path(image_path).name} -> {split}/{Path(image_save_path).name}")

        except Exception as e:
            print(f"处理{image_path}时出错: {str(e)}")


def process_drive_dataset():
    """处理DRIVE数据集"""
    print("\n开始处理DRIVE数据集...")

    for split in ['training', 'test']:
        print(f"\n处理DRIVE {split}集...")

        # 创建保存目录
        image_save_dir = os.path.join(PREPROCESSED_DIR, split, 'images_pre')
        mask_save_dir = os.path.join(PREPROCESSED_DIR, split, 'mask')
        ensure_dir(image_save_dir)
        ensure_dir(mask_save_dir)

        # 获取图像和mask路径
        image_dir = os.path.join(DRIVE_DIR, split, 'images')
        mask_dir = os.path.join(DRIVE_DIR, split, 'mask')  # 使用 mask 作为目录名

        # 获取所有图像文件
        image_paths = sorted(glob(os.path.join(image_dir, '*.tif')))

        if not image_paths:
            print(f"警告: 在{image_dir}中未找到图像文件")
            continue

        print(f"找到{len(image_paths)}张图像")

        # 处理每张图像
        for idx, image_path in enumerate(image_paths):
            try:
                # 构建mask文件名
                base_name = os.path.splitext(os.path.basename(image_path))[0]  # 保留完整的基本名称
                mask_path = os.path.join(mask_dir, f"{base_name}_mask.gif")  # 使用 _mask.gif 作为后缀

                if not os.path.exists(mask_path):
                    print(f"跳过{image_path}: 未找到对应的mask文件: {mask_path}")
                    continue

                # 加载并预处理图像
                image = load_image(image_path)
                processed_image = preprocess_image(image)
                processed_save = (processed_image * 255).astype(np.uint8)

                # 加载mask
                mask = load_mask(mask_path)

                # 保存文件
                image_save_path = os.path.join(image_save_dir, f"drive_{idx:02d}.png")
                mask_save_path = os.path.join(mask_save_dir, f"drive_{idx:02d}.png")

                if not safe_imwrite(image_save_path, processed_save):
                    print(f"保存处理后的图像失败: {image_save_path}")
                    continue

                if not safe_imwrite(mask_save_path, mask):
                    print(f"保存mask失败: {mask_save_path}")
                    continue

                print(f"已处理: {Path(image_path).name} -> {Path(image_save_path).name}")

            except Exception as e:
                print(f"处理{image_path}时出错: {str(e)}")


def process_all_data():
    """处理所有数据集"""
    print("开始数据预处理...")

    # 确保输出目录存在
    ensure_dir(PREPROCESSED_DIR)

    # 处理archive数据集
    process_archive_dataset()

    # 处理DRIVE数据集
    process_drive_dataset()

    print("\n数据预处理完成！")
    print(f"预处理后的数据保存在: {os.path.abspath(PREPROCESSED_DIR)}")


if __name__ == '__main__':
    try:
        process_all_data()
    except Exception as e:
        print(f"错误: {str(e)}")
        print("请检查数据集路径是否正确")