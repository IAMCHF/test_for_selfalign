import logging
import time

import mrcfile
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
from multiprocessing import Pool, Manager
import numpy as np
from functools import partial


def save_params_to_txt(outfile, data_list):
    with open(outfile, 'a') as params_file:
        for item in data_list:
            line_to_write = f"{item['source_path']}\t{item['temple_path']}\t"

            quaternion_str = ["{:.16f}".format(qi) for qi in item['quaternion']]
            line_to_write += "\t".join(quaternion_str)

            translation_str = ["{:.16f}".format(ti) for ti in item['translation']]
            line_to_write += "\t" + "\t".join(translation_str)

            line_to_write += "\n"
            params_file.write(line_to_write)


with mrcfile.open("EMD-3228_binned5_normalization.mrc") as mrcData:
    orig_data = mrcData.data.astype(np.float32)
temp_data_list = []
theta = np.pi / 60
phi = np.pi / 60
w = np.cos(theta / 2) * np.cos(phi / 2)
x = np.sin(theta / 2) * np.cos(phi / 2)
y = np.sin(theta / 2) * np.sin(phi / 2)
z = np.cos(theta / 2) * np.sin(phi / 2)
quaternion = [w, x, y, z]
rot_matrix = Rotation.from_quat(quaternion).as_matrix()
rotation_obj = Rotation.from_matrix(rot_matrix)
quaternion = rotation_obj.as_quat()
center = (np.array(orig_data.shape) - 1) / 2.
translation = np.random.uniform(-1.0, 1.0,
                                size=(3,))
offset = center - np.dot(rot_matrix, center) + translation
temple_mrc = affine_transform(orig_data, rot_matrix, offset=offset)

temp_data_list.append({
    'source_path': "EMD-3228_binned5_normalization.mrc",
    'temple_path': './rotated1.mrc',
    'quaternion': quaternion,
    'translation': translation,
})
with mrcfile.new('./rotated1.mrc', overwrite=True) as output_mrc:
    output_mrc.set_data(temple_mrc.astype(np.float32))
train_outfile = './train_1.txt'
train_data_list = [d for i, d in enumerate(temp_data_list)]
save_params_to_txt(train_outfile, train_data_list)


with mrcfile.open("./rotated1.mrc") as mrcData:
    orig_data = mrcData.data.astype(np.float32)
temp_data_list = []
quaternion = [-w, -x, -y, -z]
rot_matrix = Rotation.from_quat(quaternion).as_matrix()
translation = -translation
offset = center - np.dot(rot_matrix, center) + translation
temple_mrc = affine_transform(orig_data, rot_matrix, offset=offset)

temp_data_list.append({
    'source_path': "./rotated1.mrc",
    'temple_path': './rotated2.mrc',
    'quaternion': quaternion,
    'translation': translation,
})
with mrcfile.new('./rotated2.mrc', overwrite=True) as output_mrc:
    output_mrc.set_data(temple_mrc.astype(np.float32))
train_outfile = './train_2.txt'
train_data_list = [d for i, d in enumerate(temp_data_list)]
save_params_to_txt(train_outfile, train_data_list)