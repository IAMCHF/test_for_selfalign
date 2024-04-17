import os
import mrcfile
import numpy as np
from skimage.transform import rescale

def downsample_mrc(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.mrc'):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)

            # 读取原始MRC文件
            with mrcfile.open(input_filepath) as mrc_in:
                data = mrc_in.data.astype(np.float32)

            # 下采样操作
            downscaled_data = rescale(data, (32/40, 32/40, 32/40), anti_aliasing=True, mode='constant')

            # 直接写入新的MRC文件，不修改头信息
            with mrcfile.new(output_filepath, overwrite=True) as mrc_out:
                mrc_out.set_data(downscaled_data.astype(np.float32))

# 使用函数
input_folder = '/media/hao/Hard_disk_1T/datasets/gum_test_data/100-snr100'
output_folder = '/media/hao/Hard_disk_1T/datasets/gum_test_data/32/100-snr100'
downsample_mrc(input_folder, output_folder)