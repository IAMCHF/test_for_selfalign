import mrcfile
import numpy as np
from simulate import *

# 假设输入数据 shape 为 (batch_size, 50, 50, 50)
input_data_shape = (50, 50, 50)

# 创建一个与输入数据同尺寸的三维布尔掩模
mask = np.zeros(input_data_shape, dtype=np.float32)

# 球体中心坐标
center = np.array([25.0, 25.0, 25.0])

# 半径
radius = 20.0

# 创建球形区域掩模
for i in range(input_data_shape[0]):
    for j in range(input_data_shape[1]):
        for k in range(input_data_shape[2]):
            if np.linalg.norm(np.array([i, j, k]) - center) <= radius:
                mask[i, j, k] = 1.0
with mrcfile.new('mask_binned4.mrc', overwrite=True) as mrc:
    mrc.set_data(mask)

mask_wedge = generate_mask(input_data_shape)
with mrcfile.new('mask_wedge_binned4.mrc', overwrite=True) as mrc:
    mrc.set_data(mask_wedge)


# 将掩模扩展至包含 batch 维度
# mask = np.expand_dims(mask, axis=0).repeat(batch_size, axis=0)

# 接下来使用掩模与卷积相结合
# 假设 conv_layer 是一个卷积层
# 我们首先将输入数据与掩模按位相乘
# masked_data = input_data * mask

# 然后将 masked_data 输入卷积层，这里可以采用任何大小的卷积核，只要步长(stride)合适，
# 卷积后的结果就只会受到掩模为True的区域影响
# output = conv_layer(masked_data)