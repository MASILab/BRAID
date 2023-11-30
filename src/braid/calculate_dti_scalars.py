# Author: Michael Kim, Chenyu Gao
# Date: Nov 20, 2023

import subprocess
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

def run_mri_synthstrip(path_input, path_brain, path_mask, path_synthstrip_wrapper='/nfs/masi/gaoc11/Containers/synthstrip-singularity'):
    """
    Run the mri_synthstrip command-line tool to perform brain extraction on MRI images.

    Args:
        path_input (str): Path to the input MRI image.
        path_brain (str): Path to save the extracted brain image.
        path_mask (str): Path to save the binary brain mask.

    Returns:
        None
    """
    if Path(path_synthstrip_wrapper).is_file():
        subprocess.run([path_synthstrip_wrapper, '-i', path_input, '-o', path_brain, '-m', path_mask])
    else:
        print('ERROR: Cannot find the synthstrip wrapper at: {}'.format(path_synthstrip_wrapper))

def extract_b0_volume(path_dwi, path_bval, path_bvec, path_b0):
    if not (Path(path_dwi).is_file() and Path(path_bval).is_file() and Path(path_bvec).is_file()):
        raise FileNotFoundError(path_dwi)
        
    command1 = ['dwiextract', path_dwi, '-fslgrad', path_bvec, path_bval, '-', '-bzero']
    command2 = ['mrmath', '-', 'mean', path_b0, '-axis', '3']
    
    process1 = subprocess.Popen(command1, stdout=subprocess.PIPE)
    output1, _ = process1.communicate()
    
    process2 = subprocess.run(command2, input=output1, check=True)

def calculate_fa_md_maps(path_dwi, path_bval, path_bvec, path_b0, path_t1_brain_mask, path_transform_t1tob0, 
                         path_b0_brain_mask, path_b0_brain_mask_dilated, path_tensor, path_fa, path_md):
    """Generate FA, MD maps from DWI data.

    Args:
        path_dwi (str): Path to the DWI data (single shell preferred)
        path_bval (str): Path to the b-value file
        path_bvec (str): Path to the b-vector file
        path_b0 (str): Path to the b0 volume extracted from the DWI data
        path_t1_brain_mask (str): Path to the T1w brain mask
        path_transform_t1tob0 (str): Path to the ANTs transformation from T1w to b0 space 
        path_b0_brain_mask (str): Path to save the brain mask in b0 space 
        path_b0_brain_mask_dilated (str): Path to save the dilated brain mask in b0 space 
        path_tensor (str): Path to save the tensor image 
        path_fa (str): Path to save the FA map 
        path_md (str): Path to save the MD map 
    """
    
    print('Transforming the T1w brain mask to b0 space. Output: {}'.format(path_b0_brain_mask))
    subprocess.run(['mkdir', '-p', str(Path(path_b0_brain_mask).parent)])
    command = ['antsApplyTransforms', '-d', '3', '-i', path_t1_brain_mask, '-r', path_b0, '-n', 'NearestNeighbor', 
               '-t', path_transform_t1tob0, '-o', path_b0_brain_mask]
    subprocess.run(command)
    
    print('Dilating the b0 brain mask. Output: {}'.format(path_b0_brain_mask_dilated))
    command = ['fslmaths', path_b0_brain_mask, '-dilM', path_b0_brain_mask_dilated]
    subprocess.run(command)
    
    print('Calculating the tensor. Output: {}'.format(path_tensor))
    subprocess.run(['mkdir', '-p', str(Path(path_tensor).parent)])
    command = ['dwi2tensor', path_dwi, path_tensor, '-fslgrad', path_bvec, path_bval, '-mask', path_b0_brain_mask_dilated]
    subprocess.run(command)
    
    print('Calculating the FA. Output: {}'.format(path_fa))
    command = ['tensor2metric', path_tensor, '-fa', path_fa, '-mask', path_b0_brain_mask_dilated]
    subprocess.run(command)

    print('Calculating the MD. Output: {}'.format(path_md))
    command = ['tensor2metric', path_tensor, '-adc', path_md, '-mask', path_b0_brain_mask_dilated]
    subprocess.run(command)
