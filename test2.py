import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fftpack import fftn, ifftn
import mrcfile


def low_pass_gaussian_filter(image, sigma):
    f = fftn(image.astype(np.complex64))
    freq_vectors = [np.fft.fftfreq(dim_size) * dim_size for dim_size in image.shape]
    frequencies = np.stack(np.meshgrid(*freq_vectors, indexing='ij'), axis=-1)
    frequencies_norm = np.linalg.norm(frequencies, axis=-1)
    frequency_domain_filter = np.exp(-(frequencies_norm ** 2 / (2 * sigma ** 2))) / (sigma * np.sqrt(2 * np.pi))
    f *= frequency_domain_filter
    filtered_image = np.real(ifftn(f))
    return filtered_image

with mrcfile.open('IS002_291013_005_subtomo000029_binned5.mrc') as mrc:
    data = mrc.data.astype(np.float32)

sigma = 12.0

filtered_data = low_pass_gaussian_filter(data, sigma)

with mrcfile.new('IS002_291013_005_subtomo000029_binned5_low_pass.mrc', overwrite=True) as mrc:
    mrc.set_data(filtered_data)