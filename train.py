import os
import argparse
import tensorflow as tf
from datetime import datetime
import signal
import sys
import numpy as np

from model import Unet
from dataloader import DataLoader

# 添加信号处理
training_exit = False

def signal_handler(signum, frame):
    global training_exit
    print('\n捕获到 Ctrl+C，正在安全退出...')
    training_exit = True

signal.signal(signal.SIGINT, signal_handler)

def dice_coefficient(y_true, y_pred, smooth=1.0):
    """计算Dice系数"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    # 添加数值稳定性
    y_pred_f = tf.clip_by_value(y_pred_f, 1e-7, 1.0)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
    # 防止除零
    return (2. * intersection + smooth) / (union + smooth)

def binary_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, epsilon=1e-7):
    """改进的Focal Loss实现"""
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    # 计算交叉熵
    cross_entropy = -y_true * tf.math.log(y_pred)

    # 添加focal项
    weight = tf.pow(1.0 - y_pred, gamma)
    focal = alpha * weight * cross_entropy

    # 取均值
    return tf.reduce_mean(focal)

def combined_loss(y_true, y_pred):
    """改进的组合损失函数"""
    # 添加数值稳定性
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

    # 计算各个损失
    focal = binary_focal_loss(y_true, y_pred)
    dice = 1.0 - dice_coefficient(y_true, y_pred)

    # 使用更保守的权重组合
    return 0.7 * focal + 0.3 * dice

class SafeModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """安全的模型检查点保存"""
    def on_epoch_end(self, epoch, logs=None):
        try:
            super().on_epoch_end(epoch, logs)
        except Exception as e:
            print(f"\n保存检查点时发生错误: {str(e)}")

class GradientDebugCallback(tf.keras.callbacks.Callback):
    """用于调试梯度的回调"""
    def on_batch_end(self, batch, logs=None):
        if logs.get('loss') is None or np.isnan(logs['loss']):
            print(f"\n警告: 批次 {batch} 的损失为 NaN")
            # 可以在这里添加更多调试信息

"""train.py"""
# ... (previous imports remain the same)

def train_model(args):
    """训练模型"""
    try:
        # 设置内存增长
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        # 创建输出目录
        os.makedirs(args.logdir, exist_ok=True)

        # 创建数据加载器
        data_loader = DataLoader(
            batch_size=args.batch_size,
            image_size=(256, 256)
        )

        # 获取数据集
        train_dataset = data_loader.get_train_dataset(augment=True)
        val_dataset = data_loader.get_validation_dataset()

        # 创建带梯度裁剪的自定义训练步骤的模型类
        class CustomModel(Unet):
            def train_step(self, data):
                x, y = data

                with tf.GradientTape() as tape:
                    y_pred = self(x, training=True)
                    loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

                # 计算梯度
                gradients = tape.gradient(loss, self.trainable_variables)

                # 裁剪梯度
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)

                # 应用梯度
                self.optimizer.apply_gradients(zip(clipped_gradients, self.trainable_variables))

                # 更新指标
                self.compiled_metrics.update_state(y, y_pred)

                # 返回指标字典
                results = {m.name: m.result() for m in self.metrics}
                results['loss'] = loss
                return results

        # 创建模型实例
        model = CustomModel(
            data_format='channels_last',
            classes=1,
            transpose_conv=args.transpose_conv,
            name='unet_segmentation'
        )

        # 添加日志目录属性
        model.log_dir = args.logdir

        # 配置优化器（使用梯度裁剪）
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-5
        )

        # 编译模型
        model.compile(
            optimizer=optimizer,
            loss=combined_loss,
            metrics=[
                dice_coefficient,
                tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
                tf.keras.metrics.Precision(name='precision', thresholds=0.5),
                tf.keras.metrics.Recall(name='recall', thresholds=0.5)
            ]
        )

        # 构建模型
        # 创建一个示例输入来构建模型
        for x, _ in train_dataset.take(1):
            model(x, training=False)
            break

        # 创建回调函数
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_dice_coefficient',
                factor=0.2,
                patience=5,
                verbose=1,
                mode='max',
                min_lr=1e-6,
                cooldown=2
            ),
            SafeModelCheckpoint(
                filepath=os.path.join(args.logdir, 'best_model'),
                save_best_only=True,
                monitor='val_dice_coefficient',
                mode='max',
                save_weights_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_dice_coefficient',
                patience=15,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(args.logdir, 'logs'),
                write_graph=True,
                update_freq='epoch'
            ),
            GradientDebugCallback()
        ]

        # 训练模型
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1
        )

        # 如果训练正常完成，保存最终模型
        if not training_exit:
            final_model_path = os.path.join(args.logdir, 'final_model')
            model.save_weights(final_model_path)
            print(f"\n训练完成！模型权重已保存到: {final_model_path}")

        return history

    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中发生错误: {str(e)}")
        raise e
    finally:
        print("\n训练会话结束")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练U-Net分割模型')
    parser.add_argument('--logdir', type=str, default='logs', help='日志目录')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='初始学习率')  # 降低初始学习率
    parser.add_argument('--transpose_conv', action='store_true', help='使用转置卷积进行上采样')

    args = parser.parse_args()
    train_model(args)