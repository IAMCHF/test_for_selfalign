import numpy as np
import os
import mrcfile
import multiprocessing as mp

def binning_3d(file_path, output_dir,bin_size):
    with mrcfile.open(file_path) as mrc:
        data = mrc.data.astype(np.float32)

    z_dim, y_dim, x_dim = data.shape
    assert z_dim % bin_size == 0 and y_dim % bin_size == 0 and x_dim % bin_size == 0
    downsampled_data = data.reshape((z_dim // bin_size, bin_size,
                                       y_dim // bin_size, bin_size,
                                       x_dim // bin_size, bin_size)).mean(axis=(1, 3, 5))
    filename = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = filename + '_binned' + str(bin_size) + '.mrc'
    output_file = os.path.join(output_dir, output_filename)
    with mrcfile.new(output_file, overwrite=True) as mrc:
        mrc.set_data(downsampled_data)


mrc_files = [os.path.join('/media/hao/Hard_disk_1T/datasets/10045/1', f)
             for f in os.listdir('/media/hao/Hard_disk_1T/datasets/10045/1')
             if f.endswith('.mrc')]
ouput_dir = '/media/hao/Hard_disk_1T/datasets/10045/1_bin4'


with mp.Pool(processes=12) as pool:
    pool.starmap(binning_3d, [(file_path,ouput_dir, 4) for file_path in mrc_files])