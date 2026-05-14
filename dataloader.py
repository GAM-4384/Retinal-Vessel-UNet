import os
import tensorflow as tf
import numpy as np
from glob import glob


class DataLoader:
    """数据加载器类"""

    def __init__(self, data_dir=None, batch_size=4, image_size=(256, 256)):
        """初始化数据加载器"""
        # 硬编码数据目录路径
        self.base_dir = "E:/My struggle/U-Net/preprocessed"
        self.train_dir = os.path.join(self.base_dir, "training")
        self.test_dir = os.path.join(self.base_dir, "test")

        self.data_dir = data_dir if data_dir else self.base_dir
        self.batch_size = batch_size
        self.image_size = image_size

        # 验证目录结构
        self._validate_paths()

    def _validate_paths(self):
        """验证目录结构"""
        required_paths = [
            (self.train_dir, "训练集主目录"),
            (self.test_dir, "测试集主目录"),
            (os.path.join(self.train_dir, "images_pre"), "训练集图像目录"),
            (os.path.join(self.train_dir, "mask"), "训练集掩码目录"),
            (os.path.join(self.test_dir, "images_pre"), "测试集图像目录"),
            (os.path.join(self.test_dir, "mask"), "测试集掩码目录"),
        ]

        missing_paths = []
        for path, desc in required_paths:
            if not os.path.exists(path):
                missing_paths.append(f"{desc}: {path}")

        if missing_paths:
            raise ValueError("以下目录不存在：\n" + "\n".join(missing_paths))

    def _parse_image_fn(self, image_path, mask_path):
        """解析单个图像和掩码对"""

        def read_image(path):
            # 读取图像文件
            file = tf.io.read_file(path)
            # 解码PNG图像到3D张量
            image = tf.image.decode_png(file, channels=3)
            # 调整图像大小
            image = tf.image.resize(image, self.image_size)
            # 转换到[0,1]范围
            image = tf.cast(image, tf.float32) / 255.0
            # 添加标准化
            image = (image - tf.reduce_mean(image)) / (tf.math.reduce_std(image) + 1e-6)
            return image

        def read_mask(path):
            file = tf.io.read_file(path)
            # 解码PNG掩码到3D张量（单通道）
            mask = tf.image.decode_png(file, channels=1)
            # 调整掩码大小
            mask = tf.image.resize(mask, self.image_size)
            # 转换到[0,1]范围
            mask = tf.cast(mask, tf.float32) / 255.0
            return mask

        # 读取图像和掩码
        image = read_image(image_path)
        mask = read_mask(mask_path)

        return image, mask

    def _augment_data(self, image, mask):
        """数据增强"""
        # 随机左右翻转
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)

        # 随机上下翻转
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)

        # 随机调整亮度
        if tf.random.uniform(()) > 0.5:
            image = tf.image.random_brightness(image, 0.2)
            image = tf.clip_by_value(image, -1, 1)  # 由于标准化，范围改为[-1,1]

        # 随机调整对比度
        if tf.random.uniform(()) > 0.5:
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.clip_by_value(image, -1, 1)  # 由于标准化，范围改为[-1,1]

        # 随机旋转
        if tf.random.uniform(()) > 0.5:
            angle = tf.random.uniform([], -0.2, 0.2)  # ±0.2弧度约等于±11.5度
            image = tf.image.rot90(image, k=tf.cast(angle * 2 / np.pi, tf.int32))
            mask = tf.image.rot90(mask, k=tf.cast(angle * 2 / np.pi, tf.int32))

        return image, mask

    def get_dataset(self, split='train', augment=False):
        """获取数据集"""
        # 根据split选择正确的目录
        data_dir = self.train_dir if split == 'train' else self.test_dir

        # 构建图像和掩码的路径
        images_path = os.path.join(data_dir, 'images_pre')
        masks_path = os.path.join(data_dir, 'mask')

        # 获取所有图像和掩码文件
        image_files = sorted(glob(os.path.join(images_path, '*.*')))
        mask_files = sorted(glob(os.path.join(masks_path, '*.*')))

        if not image_files or not mask_files:
            raise ValueError(
                f"未找到图像或掩码文件。\n"
                f"图像路径: {images_path} (找到 {len(image_files)} 个文件)\n"
                f"掩码路径: {masks_path} (找到 {len(mask_files)} 个文件)"
            )

        if len(image_files) != len(mask_files):
            raise ValueError(
                f"图像和掩码数量不匹配。\n"
                f"图像数量: {len(image_files)}\n"
                f"掩码数量: {len(mask_files)}"
            )

        # 创建数据集
        dataset = tf.data.Dataset.from_tensor_slices((image_files, mask_files))

        # 打乱数据（仅在训练时）
        if split == 'train':
            dataset = dataset.shuffle(buffer_size=len(image_files), reshuffle_each_iteration=True)

        # 映射解析函数
        dataset = dataset.map(
            lambda x, y: tf.numpy_function(
                func=self._parse_image_fn,
                inp=[x, y],
                Tout=[tf.float32, tf.float32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # 设置张量形状
        dataset = dataset.map(
            lambda x, y: (
                tf.ensure_shape(x, [*self.image_size, 3]),
                tf.ensure_shape(y, [*self.image_size, 1])
            )
        )

        # 数据增强
        if augment:
            dataset = dataset.map(
                self._augment_data,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # 批次化和预取
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def get_test_dataset(self):
        """获取测试数据集"""
        return self.get_dataset(split='test', augment=False)

    def get_train_dataset(self, augment=True):
        """获取训练数据集"""
        return self.get_dataset(split='train', augment=augment)

    def get_validation_dataset(self):
        """获取验证数据集（使用测试集）"""
        return self.get_test_dataset()

    def get_dataset_info(self):
        """获取数据集信息"""
        train_images = glob(os.path.join(self.train_dir, 'images_pre', '*.*'))
        train_masks = glob(os.path.join(self.train_dir, 'mask', '*.*'))
        test_images = glob(os.path.join(self.test_dir, 'images_pre', '*.*'))
        test_masks = glob(os.path.join(self.test_dir, 'mask', '*.*'))

        return {
            'train_size': len(train_images),
            'train_masks': len(train_masks),
            'test_size': len(test_images),
            'test_masks': len(test_masks),
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'train_dir': self.train_dir,
            'test_dir': self.test_dir
        }


# 使用示例
if __name__ == '__main__':
    # 初始化数据加载器
    data_loader = DataLoader(batch_size=4, image_size=(256, 256))

    # 打印数据集信息
    info = data_loader.get_dataset_info()
    print("\n数据集信息:")
    for key, value in info.items():
        print(f"{key}: {value}")

    # 测试加载一个批次
    print("\n测试加载数据...")
    try:
        train_dataset = data_loader.get_train_dataset()
        for images, masks in train_dataset.take(1):
            print(f"批次形状 - 图像: {images.shape}, 掩码: {masks.shape}")
            print("数据加载测试成功！")
            break
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")