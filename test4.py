import numpy as np
from scipy.spatial.transform import Rotation
import mrcfile
from scipy.ndimage import affine_transform
from scipy.stats import special_ortho_group


def random_rotate_mrc(orig_data):
    rot_matrix = special_ortho_group.rvs(3)
    quaternion = Rotation.from_matrix(rot_matrix).as_quat()
    center = (np.array(orig_data.shape) - 1) / 2.
    offset = center - np.dot(rot_matrix, center)
    rotated_data = affine_transform(orig_data, rot_matrix, offset=offset)
    return quaternion, offset, rotated_data


def get_cubes_one(ori, temple_mrc, quaternion, translation, start=0):
    with mrcfile.new('{}/rotated_{}.mrc'.format("/home/hao/PycharmProjects/test", start), overwrite=True) as output_mrc:
        output_mrc.set_data(temple_mrc.astype(np.float32))
    save_params_to_txt('{}/params_{}.txt'.format("/home/hao/PycharmProjects/test", start), ori,
                       '{}/rotated_{}.mrc'.format("/home/hao/PycharmProjects/test", start), quaternion, translation)
    return 0


def get_cubes(start):
    mrc = "/home/hao/PycharmProjects/test/IS002_291013_005_subtomo000002.mrc"
    with mrcfile.open(mrc) as mrcData:
        orig_data = mrcData.data.astype(np.float32)
    rotation_quaternion, saved_translation, temple_mrc = random_rotate_mrc(orig_data)
    get_cubes_one(mrc, temple_mrc, rotation_quaternion, saved_translation, start=start)


def params_to_matrix(quaternion):
    if isinstance(quaternion, np.ndarray):  # 检查是否是NumPy数组
        quaternion = Rotation.from_quat(quaternion)  # 转换为Rotation对象
    # translation_vector = np.array(particle_center) + translation
    # 将四元数转换为旋转矩阵
    rot_matrix = Rotation.from_quat(quaternion)
    return rot_matrix


def save_params_to_txt(outfile, source_path, temple_path, quaternion, translation):
    with open(outfile, 'a') as params_file:
        line_to_write = f"{source_path}\t{temple_path}\t"

        # 将四元数转换为字符串，保留高精度
        quaternion_str = ["{:.16f}".format(qi) for qi in quaternion]
        line_to_write += "\t".join(quaternion_str)

        # 将平移向量转换为字符串，保留高精度
        translation_str = ["{:.16f}".format(ti) for ti in translation]
        line_to_write += "\t" + "\t".join(translation_str)

        line_to_write += "\n"
        params_file.write(line_to_write)


get_cubes(4)
