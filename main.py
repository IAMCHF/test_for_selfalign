# This is a sample Python script.
import mrcfile
import warnings
import sys
import numpy as np
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, Dense, Concatenate, Lambda, Activation, BatchNormalization, \
    Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU

warnings.simplefilter('ignore')  # to mute some warnings produced when opening the tomos
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def mw2d(dim, missingAngle=[30, 30]):
    mw = np.zeros((dim, dim), dtype=np.double)
    missingAngle = np.array(missingAngle)
    missing = np.pi / 180 * (90 - missingAngle)  # [1.04719755 1.04719755], 是[60°,60°]的弧度制
    for i in range(dim):
        for j in range(dim):
            y = (i - dim / 2)
            x = (j - dim / 2)
            if x == 0:  # and y!=0:
                theta = np.pi / 2
            # elif x==0 and y==0:
            #    theta=0
            # elif x!=0 and y==0:
            #    theta=np.pi/2
            else:
                theta = abs(np.arctan(y / x))

            if x ** 2 + y ** 2 <= min(dim / 2, dim / 2) ** 2:
                if x > 0 and y > 0 and theta < missing[0]:
                    mw[i, j] = 1  # np.cos(theta)
                if x < 0 and y < 0 and theta < missing[0]:
                    mw[i, j] = 1  # np.cos(theta)
                if x > 0 and y < 0 and theta < missing[1]:
                    mw[i, j] = 1  # np.cos(theta)
                if x < 0 and y > 0 and theta < missing[1]:
                    mw[i, j] = 1  # np.cos(theta)

            if int(y) == 0:
                mw[i, j] = 1
    # from mwr.util.image import norm_save
    # norm_save('mw.tif',self._mw)
    return mw


# Press the green button in the gutter to run the script.

def get_model(source_volume, template_volume):
    inputs = tf.concat([source_volume, template_volume], axis=-2)
    batch_size = tf.shape(inputs)[0]

    input_image = Lambda(lambda x: tf.expand_dims(x, -1))(inputs)

    net = Conv3D(64, (1, 3, 3), padding='valid', strides=(1, 1, 1))(input_image)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)

    net = Conv3D(64, (1, 1, 1), padding='valid', strides=(1, 1, 1))(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)

    net = Conv3D(64, (1, 1, 1), padding='valid', strides=(1, 1, 1))(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)

    net = Conv3D(128, (1, 1, 1), padding='valid', strides=(1, 1, 1))(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)

    net = Conv3D(1024, (1, 1, 1), padding='valid', strides=(1, 1, 1))(net)
    net = BatchNormalization()(net)

    # Symmetric function: max pooling
    net = Lambda(lambda x: tf.reduce_max(x, axis=1, keepdims=True))(net)
    net = Lambda(lambda x: tf.reshape(x, (batch_size, -1)))(net)

    source_global_feature = Lambda(lambda x: x[:, :int(batch_size / 2), :])(net)
    template_global_feature = Lambda(lambda x: x[:, int(batch_size / 2):, :])(net)
    # 新增网络结构预测旋转和平移参数
    concatenated_features = Concatenate(axis=-1)([source_global_feature, template_global_feature])

    rotation_pred = Dense(512, activation='relu')(concatenated_features)
    rotation_pred = Dropout(0.3)(rotation_pred)
    rotation_pred = Dense(256, activation='relu')(rotation_pred)
    rotation_output = Dense(4, activation='tanh')(rotation_pred)  # 四元数预测，范围-1到1之间

    translation_pred = Dense(3)(concatenated_features)
    translation_output = LeakyReLU(alpha=0.2)(translation_pred)

    print('rotation_output', rotation_output)
    print('translation_output', translation_output)


def create_3d_gaussian_kernel(sigma, target_shape):
    # 确保给定的形状是一个立方体
    assert len(set(target_shape)) == 1, "Gaussian kernel shape should be cubic."

    half_shape = target_shape[0] // 2
    grid = np.arange(-half_shape, half_shape + 1)
    kernel_1d = np.exp(-(grid ** 2 / (2 * sigma ** 2)))
    kernel_3d = np.outer(kernel_1d, kernel_1d[:, np.newaxis]) * kernel_1d[np.newaxis, :, np.newaxis]
    kernel_3d /= np.sum(kernel_3d)  # 归一化以便保持能量守恒

    # 确保高斯核的实际形状与目标形状一致
    kernel_3d = kernel_3d[(slice(None),) * (len(target_shape) - kernel_3d.ndim) + (-half_shape, half_shape)]
    kernel_3d = kernel_3d.reshape(target_shape)

    return kernel_3d


def low_pass_filter(input_mrc_path, output_mrc_path, padding_size=0, sigma=2):
    if padding_size <= 0:
        raise ValueError("Padding size must be greater than 0.")

    # 读取MRC文件并转换为float32类型
    with mrcfile.open(input_mrc_path) as mrc:
        original_data = mrc.data.astype(np.complex64)  # 将数据类型改为复数，因为傅里叶变换结果为复数

    # 添加零填充
    padded_shape = tuple(d + 2 * padding_size for d in original_data.shape)
    padded_data = np.zeros(padded_shape, dtype=np.complex64)
    padded_data[padding_size:-padding_size,
    padding_size:-padding_size,
    padding_size:-padding_size] = original_data

    # 对填充后数据进行3D傅里叶变换
    padded_data_fft = np.fft.fftn(padded_data, axes=(0, 1, 2))

    # 创建与填充后数据同样大小的3D高斯核
    gaussian_kernel = create_3d_gaussian_kernel(sigma, padded_shape)

    # 确保高斯核已成功创建
    assert gaussian_kernel is not None, "Failed to create Gaussian kernel."

    # 应用高斯核到频域数据上
    filtered_fft = padded_data_fft * gaussian_kernel

    # 进行逆傅里叶变换得到滤波后的图像
    filtered_data = np.fft.ifftn(filtered_fft, axes=(0, 1, 2)).real

    # 裁剪回原始大小
    filtered_data = filtered_data[
                    padding_size:-padding_size,
                    padding_size:-padding_size,
                    padding_size:-padding_size
                    ]

    # 输出滤波后的图像至新的MRC文件
    with mrcfile.new(output_mrc_path, overwrite=True) as mrc:
        mrc.set_data(filtered_data.astype(np.float32))



if __name__ == '__main__':
    low_pass_filter("./IS002_291013_005_subtomo000002.mrc", "./output_filtered.mrc", padding_size=10)
    # with mrcfile.open("./IS002_291013_005_subtomo000002.mrc") as mrcData:
    #     ow_data = mrcData.data.astype(np.float32) * -1
    # iw_data = ow_data
    # data = tf.concat([iw_data, ow_data], axis=0)
    # print(type(iw_data))
    # print(type(data))
    # print(data.shape[0])
    #
    # get_model(np.array(iw_data), np.array(ow_data))

    # print("ow_data", type(ow_data), ow_data)
    # data = np.rot90(ow_data, k=1, axes=(0, 1))  # clock wise of counter clockwise逆时针??
    # print("data",data)
    # print("data.shape[1]", data.shape[1])
    # mw = mw2d(data.shape[1])
    # ld1 = 1
    # ld2 = 0
    # mw = mw * ld1 + (1 - mw) * ld2
    # print("mw", type(mw), mw)
    # outData = np.zeros(data.shape, dtype=np.float32)
    # mw_shifted = np.fft.fftshift(mw)
    # print("mw_shifted", mw_shifted)
    # for i, item in enumerate(data):
    #     print("i = ", i)
    #     print("item = ", item)
    #     outData_i = np.fft.ifft2(mw_shifted * np.fft.fft2(item))
    #     outData[i] = np.real(outData_i)
    #
    # outData.astype(np.float32)
    # outData = np.rot90(outData, k=3, axes=(0, 1))
    # print("outData", type(outData), outData)

    # missingAngle = [30, 30]
    # missingAngle = np.array(missingAngle)
    # print("missingAngle", type(missingAngle), missingAngle)
    # print("90 - missingAngle", type(90 - missingAngle), 90 - missingAngle)
    # missing = np.pi / 180 * (90 - missingAngle)
    # print(type(missing), missing)
    # tomo = np.mat([[1.2, 1.5],
    #      [2.6, 2.4],
    #      [4.2, 6.2],
    #      [2.0, 5.2],
    #      [8.2, 1.2],
    #      [6.2, 3.2],
    #      [2.2, 4.2]])
    # sp = np.array(tomo.shape)
    # print(sp)
    # sp2 = sp // 2
    # print(sp2)
    # bintomo = resize(tomo, sp2, anti_aliasing=True)
    # deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(a, axis=0))[:-1]
    # weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
    # weights[-1] = 0
    #
    # # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
    # weights = weights / weights.sum()
    # print(weights)
    # sum = 0.0
    # for i in weights:
    #     sum += i
    # print(sum)
    # with mrcfile.open('/media/hao/Sata500g/dataset/shrec21_full_dataset/model_0/reconstruction.mrc', permissive=True) as tomo0:
    #     tomo0_data = tomo0.data
    #     print(tomo0_data.shape)
    #     print(tomo0_data.shape[1])
    #     print(tomo0_data[:, 0])
    # X = np.array([[0, 1], [2, 3], [4, 5]])
    # print(X.shape)
    # print(X.shape[1])
    # print(X[:, 0])  # x[:,n]表示在全部数组（维）中取第n个数据
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
