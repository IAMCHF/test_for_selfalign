import mrcfile
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
from multiprocessing import Pool, Manager
import numpy as np

with mrcfile.open("EMD-3228_binned5_normalization.mrc") as mrcData:
    orig_data = mrcData.data.astype(np.float32)

euler_angles = np.array([-2.3, -2.4, -2.9], np.float32)
# translation = np.array([-3.1790831524895404, -3.2182084280771264, 3.4852564365899639], np.float32)
rotation_obj = Rotation.from_euler('zyz', euler_angles, degrees=False)
rot_matrix = rotation_obj.as_matrix()
center = (np.array(orig_data.shape) - 1) / 2.

offset = center - np.dot(rot_matrix, center)

# offset = center - np.dot(rot_matrix, center) + translation
temple_mrc = affine_transform(orig_data, rot_matrix, offset=offset)
print("euler_angles", euler_angles)
# print("translation", translation)
with mrcfile.new('./rotated1.mrc', overwrite=True) as output_mrc:
    output_mrc.set_data(temple_mrc.astype(np.float32))

with mrcfile.open("./rotated1.mrc") as mrcData:
    orig_data = mrcData.data.astype(np.float32)
euler_angles = np.array([2.9, 2.4, 2.3], np.float32)
# translation = -translation
rotation_obj = Rotation.from_euler('zyz', euler_angles, degrees=False)
rot_matrix = rotation_obj.as_matrix()

offset = center - np.dot(rot_matrix, center)

# offset = center - np.dot(rot_matrix, center) + translation
temple_mrc = affine_transform(orig_data, rot_matrix, offset=offset)
print("euler_angles", euler_angles)
# print("translation", translation)
with mrcfile.new('./rotated2.mrc', overwrite=True) as output_mrc:
    output_mrc.set_data(temple_mrc.astype(np.float32))
