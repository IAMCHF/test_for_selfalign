from scipy.stats import zscore
from tensorflow.keras.utils import Sequence
import numpy as np
import mrcfile
import os


def normalize_z_score(mrc_path):
    with mrcfile.open(mrc_path, permissive=True) as mrcData:
        mrc_data = mrcData.data.astype(np.float32)
    normalized_data = zscore(mrc_data, axis=None)
    normalized_data = normalized_data.reshape(mrc_data.shape)
    return normalized_data


class PCRNetDataSequence(Sequence):
    def __init__(self, x_files, batch_size):
        self.x_files = x_files
        self.batch_size = batch_size
        self.perm = np.random.permutation(len(self.x_files))

    def __len__(self):
        return int(np.ceil(len(self.x_files) / float(self.batch_size)))

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.x_files))

    def __getitem__(self, i):
        idx = slice(i * self.batch_size, (i + 1) * self.batch_size)
        idx = self.perm[idx]

        rx = np.array([self.load_mrc_data(self.x_files[j]) for j in idx])

        return rx

    @staticmethod
    def load_mrc_data(file_path):
        return normalize_z_score(file_path)

    @staticmethod
    def load_rotation_translation(params_file_path):
        with open(params_file_path, 'r') as f:
            line = f.readline().strip()
            ori_path, temple_mrc_path, quaternion_str, translation_str = line.split('\t')[0:4]
            quaternion = [float(qi) for qi in quaternion_str.split('\t')]
            translation = [float(ti) for ti in translation_str.split('\t')]
            return quaternion, translation


def prepare_pcrnet_dataseq(data_folder, batch_size, particle_radius=75):
    train_data = PCRNetDataSequence(["./IS002_291013_005_subtomo000002.mrc"], 1)
    print(type(train_data))
    print(train_data)
