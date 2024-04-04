import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fftpack import fftn, ifftn
import mrcfile


def low_pass_gaussian_filter(image, sigma):
    """
    使用高斯滤波器在频域进行3D低通滤波
    参数:
    image: 3D numpy数组，原始图像数据
    sigma: 浮点数，高斯滤波器的标准差，控制滤波强度
    返回:
    filtered_image: 3D numpy数组，经过低通滤波后的图像
    """
    # 将图像转换至频域
    f = fftn(image.astype(np.complex64))  # 确保数据类型为复数

    # 计算每个维度的频率向量
    freq_vectors = [np.fft.fftfreq(dim_size) * dim_size for dim_size in image.shape]

    # 构建3D频率网格并计算每个频率点的距离（欧氏距离）
    frequencies = np.stack(np.meshgrid(*freq_vectors, indexing='ij'), axis=-1)
    frequencies_norm = np.linalg.norm(frequencies, axis=-1)

    # 创建3D高斯核，用于在频域应用滤波
    frequency_domain_filter = np.exp(-(frequencies_norm ** 2 / (2 * sigma ** 2)))

    # 直接在频域应用3D高斯滤波器
    f *= frequency_domain_filter

    # 将结果从频域转换回空域
    filtered_image = np.real(ifftn(f))

    return filtered_image


# 打开并读取MRC文件
with mrcfile.open('IS002_291013_005_subtomo000002.mrc') as mrc:
    data = mrc.data.astype(np.float32)

# 设置高斯滤波器的标准差以控制滤波强度
sigma = 10.0  # 根据实际需求调整此值

# 对图像进行低通滤波
filtered_data = low_pass_gaussian_filter(data, sigma)

# 将处理后的数据保存为新的MRC文件
with mrcfile.new('output3.mrc', overwrite=True) as mrc:
    mrc.set_data(filtered_data)

print("已完成低通滤波并将结果保存为 'output.mrc' 文件。")