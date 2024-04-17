import mrcfile
import numpy as np

from simulate import apply_wedge

with mrcfile.open("mask_wedge_binned5.mrc", permissive=True) as mrc:
    mask_wedge_binned5 = mrc.data.astype(np.float32)
with mrcfile.open("emd3228binned5_normalization.mrc", permissive=True) as mrc:
    emd3228binned5_normalization = mrc.data.astype(np.float32)
applied_wedge = apply_wedge(emd3228binned5_normalization, mask_wedge_binned5)
with mrcfile.new('applied_wedge.mrc', overwrite=True) as output_mrc:
    output_mrc.set_data(applied_wedge.astype(np.float32))
