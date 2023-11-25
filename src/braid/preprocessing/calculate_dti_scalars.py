#
# Author: Michael Kim, Chenyu Gao
# Date: Nov 20, 2023

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import pdb


def extract_single_shell(path_dwi_in, path_bval_in, path_bvec_in, path_dwi_out, path_bval_out, path_bvec_out, threshold=1500):
    """
    Extracts a single shell from a DWI dataset based on a given threshold.

    Args:
        path_dwi_in (str): Path to the input DWI dataset.
        path_bval_in (str): Path to the input b-value file.
        path_bvec_in (str): Path to the input b-vector file.
        path_dwi_out (str): Path to save the output DWI dataset.
        path_bval_out (str): Path to save the output b-value file.
        path_bvec_out (str): Path to save the output b-vector file.
        threshold (float, optional): Threshold value to determine the volumes to extract. Defaults to 1500.

    Returns:
        None
    """
    
    # Load the input DWI data
    nii_dwi_in = nib.load(path_dwi_in)
    data_dwi_in = nii_dwi_in.get_fdata()
    
    bval_in = np.loadtxt(path_bval_in)  # (#volumes,)
    bvec_in = np.loadtxt(path_bvec_in)  # (3, #volumes)

    # Get indices of volumes to extract
    indices = np.where(bval_in < threshold)
    bval_new = bval_in[indices]
    bvec_new = np.stack([dir[indices] for dir in bvec_in], axis=0)  # (3, #volumes_singleshell)

    # Generate txt for bval and bvec (replace 0.0 with "0")
    bval_str = []
    for bval in bval_new:
        if bval == 0:
            bval_str.append('0')
        else:
            bval_str.append(str(bval))
    bval_txt = ' '.join(bval_str)
    bval_txt = bval_txt + '\n'
    
    bvec_txt = ''
    for dir in bvec_new:
        ndir = []
        for d in dir:
            if d == 0:
                ndir.append('0')
            else:
                ndir.append(str(d))
        s = '  '.join(ndir)
        s += '\n'
        bvec_txt += s

    print('Saving the single-shell bvec to: {}'.format(path_bvec_out))
    with open(path_bvec_out, 'w') as f:
        f.write(bvec_txt)
    print('Saving the single-shell bval to: {}'.format(path_bval_out))
    with open(path_bval_out, 'w') as f:
        f.write(bval_txt)

    print('Saving the single-shell NIFTI to: {}'.format(path_dwi_out))
    data_dwi_out = data_dwi_in[:, :, :, bval_in < threshold]
    nii_dwi_out = nib.Nifti1Image(data_dwi_out, nii_dwi_in.affine)
    nib.save(nii_dwi_out, path_dwi_out)


# test
path_dwi_in = '/nfs2/harmonization/BIDS/HCPA/derivatives/sub-HCA6439780/PreQual/PREPROCESSED/dwmri.nii.gz'
path_bval_in = '/nfs2/harmonization/BIDS/HCPA/derivatives/sub-HCA6439780/PreQual/PREPROCESSED/dwmri.bval'
path_bvec_in = '/nfs2/harmonization/BIDS/HCPA/derivatives/sub-HCA6439780/PreQual/PREPROCESSED/dwmri.bvec'
path_dwi_out = '/nfs/masi/gaoc11/projects/BRAID/data/test_preproc/dwmri.nii.gz'
path_bval_out = '/nfs/masi/gaoc11/projects/BRAID/data/test_preproc/dwmri.bval'
path_bvec_out = '/nfs/masi/gaoc11/projects/BRAID/data/test_preproc/dwmri.bvec'

extract_single_shell(path_dwi_in, path_bval_in, path_bvec_in, path_dwi_out, path_bval_out, path_bvec_out, threshold=1500)

# =============================================================================

