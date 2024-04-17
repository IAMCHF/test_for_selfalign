import mrcfile
import numpy as np

with mrcfile.open('mask_binned5.mrc') as mrc:
    mask_binned5 = mrc.data.astype(np.float32)

with mrcfile.new('mask_binned5.mrc', overwrite=True) as mrc:
    mrc.set_data(mask_binned5)

with mrcfile.open('observed_mask.mrc') as mrc:
    data = mrc.data.astype(np.float32)
    with mrcfile.new('observed_mask1.mrc', overwrite=True) as mrc:
        mrc.set_data(data * mask_binned5)