import numpy as np
import mrcfile
from scipy.fftpack import fftn, ifftn, fftshift

# 读取mrc文件
# with mrcfile.open('IS002_291013_005_subtomo000002.mrc') as mrc:
#     data = mrc.data.astype(np.float32)
#
# # 进行低通滤波，假设cutoff_freq是一个可调节的参数
# cutoff_freq = 2  # 可以根据实际需求调整这个值
# filtered_data = low_pass_filter(data, cutoff_freq)
#
# 将处理后的数据保存为mrc文件
# with mrcfile.new('output.mrc', overwrite=True) as mrc:
#     mrc.set_data(filtered_data)
# print("当cutoff_freq等于0时，不会对图像进行低通滤波。")

# data_fft = np.fft.fftn(data)
# data_fft_shifted = np.fft.fftshift(data_fft)
# amplitude_spectrum = np.abs(data_fft_shifted)
# # amplitude_spectrum = amplitude_spectrum / np.max(amplitude_spectrum)
# with mrcfile.new('data_fft.mrc', overwrite=True) as output_mrc:
#     output_mrc.set_data(data_fft.astype(np.float32))
# with mrcfile.new('data_fft_shifted.mrc', overwrite=True) as output_mrc:
#     output_mrc.set_data(data_fft_shifted.astype(np.float32))
# with mrcfile.new('amplitude_spectrum.mrc', overwrite=True) as output_mrc:
#     output_mrc.set_data(amplitude_spectrum.astype(np.float32))
#
# filtered_data_fft = np.fft.fftn(filtered_data)
# filtered_data_fft_shifted = np.fft.fftshift(filtered_data_fft)
# filtered_amplitude_spectrum = np.abs(filtered_data_fft_shifted)
# with mrcfile.new('filtered_data_fft.mrc', overwrite=True) as output_mrc:
#     output_mrc.set_data(filtered_data_fft.astype(np.float32))
# with mrcfile.new('filtered_data_fft_shifted.mrc', overwrite=True) as output_mrc:
#     output_mrc.set_data(filtered_data_fft_shifted.astype(np.float32))
# with mrcfile.new('filtered_amplitude_spectrum.mrc', overwrite=True) as output_mrc:
#     output_mrc.set_data(filtered_amplitude_spectrum.astype(np.float32))
#
from simulate import *

mask = generate_mask((32, 32, 32))
# mask_fft = np.fft.fftn(mask)
# mask_shifted = np.fft.fftshift(mask_fft)
# mask_amplitude_spectrum = np.abs(mask_shifted)
with mrcfile.new('mask_wedge_32.mrc', overwrite=True) as mrc:
    mrc.set_data(mask)
# with mrcfile.new('mask_fft.mrc', overwrite=True) as mrc:
#     mrc.set_data(mask_fft)
# with mrcfile.new('mask_shifted.mrc', overwrite=True) as mrc:
#     mrc.set_data(mask_shifted)
# with mrcfile.new('mask_amplitude_spectrum.mrc', overwrite=True) as mrc:
#     mrc.set_data(mask_amplitude_spectrum)
