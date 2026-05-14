import tensorflow as tf
from tensorflow.keras import layers


class _EncodeBlock(tf.keras.Model):
    """基础编码器块"""

    def __init__(self, filters, stage, data_format):
        super(_EncodeBlock, self).__init__(name=f'encode_block_{stage}')

        filters1, filters2 = filters
        conv_name_base = f'encode{stage}_conv'
        bn_name_base = f'encode{stage}_bn'
        pool_name = f'encode{stage}_pool'
        bn_axis = 1 if data_format == 'channels_first' else 3

        # 第一个卷积块
        self.conv2a = layers.Conv2D(
            filters1, (3, 3),
            padding='same',
            kernel_initializer='he_normal',
            data_format=data_format,
            name=f'{conv_name_base}2a'
        )
        self.bn2a = layers.BatchNormalization(
            axis=bn_axis,
            name=f'{bn_name_base}2a'
        )

        # 第二个卷积块
        self.conv2b = layers.Conv2D(
            filters2, (3, 3),
            padding='same',
            kernel_initializer='he_normal',
            data_format=data_format,
            name=f'{conv_name_base}2b'
        )
        self.bn2b = layers.BatchNormalization(
            axis=bn_axis,
            name=f'{bn_name_base}2b'
        )

        self.pool = layers.MaxPooling2D(data_format=data_format, name=pool_name)

    def call(self, input_tensor, training=True):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        # 添加梯度检查
        tf.debugging.check_numerics(x, "Encoder output has inf/nan values")

        poolx = self.pool(x)
        return x, poolx


class _DecodeBlock(tf.keras.Model):
    """基础解码器块"""

    def __init__(self, filters, stage, data_format, transpose_conv=False):
        super(_DecodeBlock, self).__init__(name=f'decode_block_{stage}')

        filters1, filters2, filter3 = filters
        self.transpose_conv = transpose_conv

        conv_name_base = f'decode{stage}_conv'
        bn_name_base = f'decode{stage}_bn'
        up_name_base = f'decode{stage}_'
        bn_axis = 1 if data_format == 'channels_first' else 3

        # 卷积层
        self.conv2a = layers.Conv2D(
            filters1, (3, 3),
            padding='same',
            kernel_initializer='he_normal',
            data_format=data_format,
            name=f'{conv_name_base}2a'
        )
        self.bn2a = layers.BatchNormalization(
            axis=bn_axis,
            name=f'{bn_name_base}2a'
        )

        self.conv2b = layers.Conv2D(
            filters2, (3, 3),
            padding='same',
            kernel_initializer='he_normal',
            data_format=data_format,
            name=f'{conv_name_base}2b'
        )
        self.bn2b = layers.BatchNormalization(
            axis=bn_axis,
            name=f'{bn_name_base}2b'
        )

        # 上采样层
        if self.transpose_conv:
            self.up = layers.Conv2DTranspose(
                filter3, (2, 2),
                strides=(2, 2),
                padding='same',
                kernel_initializer='he_normal',
                data_format=data_format,
                name=f'{up_name_base}transpose'
            )
        else:
            self.up = layers.UpSampling2D(
                size=(2, 2),
                data_format=data_format,
                name=f'{up_name_base}upsample'
            )

    def call(self, input_tensor, skip_connection, training=True):
        # 上采样
        x = self.up(input_tensor)

        # 添加梯度检查
        tf.debugging.check_numerics(x, "Upsampling output has inf/nan values")

        x = self.conv2a(x)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        # 添加梯度检查
        tf.debugging.check_numerics(x, "Decoder output has inf/nan values")

        return x


class Unet(tf.keras.Model):
    """基础U-Net模型"""

    def __init__(self, data_format, classes, transpose_conv=False, name=''):
        super(Unet, self).__init__(name=name)

        if data_format not in ('channels_first', 'channels_last'):
            raise ValueError('data_format must be channels_first or channels_last')

        self.concat_axis = 3 if data_format == 'channels_last' else 1

        # 编码器路径
        self.e1 = _EncodeBlock([32, 32], stage=1, data_format=data_format)
        self.e2 = _EncodeBlock([64, 64], stage=2, data_format=data_format)
        self.e3 = _EncodeBlock([128, 128], stage=3, data_format=data_format)
        self.e4 = _EncodeBlock([256, 256], stage=4, data_format=data_format)

        # 解码器路径
        self.d4 = _DecodeBlock([512, 512, 256], stage=4, data_format=data_format, transpose_conv=transpose_conv)
        self.d3 = _DecodeBlock([256, 256, 128], stage=3, data_format=data_format, transpose_conv=transpose_conv)
        self.d2 = _DecodeBlock([128, 128, 64], stage=2, data_format=data_format, transpose_conv=transpose_conv)
        self.d1 = _DecodeBlock([64, 64, 32], stage=1, data_format=data_format, transpose_conv=transpose_conv)

        # 输出层
        self.final_conv = layers.Conv2D(
            classes, (1, 1),
            kernel_initializer='he_normal',
            data_format=data_format,
            activation='sigmoid',  # 添加sigmoid激活函数
            name='conv_output'
        )

    def call(self, inputs, training=True):
        # 编码器路径
        e1x, x = self.e1(inputs, training=training)
        e2x, x = self.e2(x, training=training)
        e3x, x = self.e3(x, training=training)
        e4x, x = self.e4(x, training=training)

        # 解码器路径
        x = self.d4(x, e4x, training=training)
        x = layers.concatenate([x, e4x], axis=self.concat_axis)

        x = self.d3(x, e3x, training=training)
        x = layers.concatenate([x, e3x], axis=self.concat_axis)

        x = self.d2(x, e2x, training=training)
        x = layers.concatenate([x, e2x], axis=self.concat_axis)

        x = self.d1(x, e1x, training=training)
        x = layers.concatenate([x, e1x], axis=self.concat_axis)

        x = self.final_conv(x)

        # 添加梯度检查
        tf.debugging.check_numerics(x, "Final output has inf/nan values")

        return x