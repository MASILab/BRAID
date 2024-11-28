# Date: Nov 26, 2024

import argparse
import subprocess
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import braid.registrations
import braid.calculate_dti_scalars

def convert_to_brain_mask(path_input, path_output, background=0):
    """Convert the segmentation image to a binary brain mask.

    Args:
        path_input (str): path to the segmentation image.
        path_output (str): path to the output brain mask.
    """
    img = nib.load(path_input)
    data = img.get_fdata()
    brain_mask = (data != background).astype(int)
    nib.save(nib.Nifti1Image(brain_mask, img.affine), path_output)

def apply_skull_strip_mask(path_input, path_mask, path_output):
    """Apply a binary brain mask to an input image. The input and the mask should be in the same space.
    We assume the binary mask is 0 for background, and 1 for brain, and there is no other value in the mask.

    Args:
        path_input (str): Path to the input image
        path_mask (str): Path to the brain mask
        path_output (str): Path to save the output image
    """
    
    command = ['fslmaths', str(path_input), '-mul', str(path_mask), str(path_output)]
    subprocess.run(command)

def main():
    parser = argparse.ArgumentParser(
        prog='BRAID inference for one sample',
        description=(
            'Run BRAID model inference for one sample.' 
            'It performs the following steps in sequence: '
            '1) Calculation of FA and MD images; '
            '2) Registrations, between T1 and template, and between b0 and T1; '
            '3) Transform the FA and MD images to the template space (affine or non-rigid); '
            '4) Run five independently trained ResNet models obtained through five-fold cross-validation, '
            'for WM age nonrigid, WM age affine, and GM age ours, respectively. '
            'In the end, for each brain age estimation type, '
            'it provides five individual estimations, mean, and '
            'standard deviation of the estimations in a single-row csv file.'
        )
    )
    parser.add_argument('-d', '--dwi', type=str, required=True, help='path to the NIFTI file of the diffusion-weighted MR image.')
    parser.add_argument('-v', '--bval', type=str, required=True, help='path to the b value table (typically a file ending with ".bval") for the diffusion-weighted MR image.')
    parser.add_argument('-c', '--bvec', type=str, required=True, help='path to the b vector table (typically a file ending with "*.bvec") for the diffusion-weighted MR image.')
    parser.add_argument('-t', '--t1w', type=str, required=True, help='path to the NIFTI file of the T1-weighted MR image of the same subject, preferably acquired in the same imaging session as the diffusion-weighted MR image.')
    parser.add_argument('-tm', '--t1w_mask', type=str, required=True, help='path to the NIFTI file of the brain mask for the T1-weighted MR image. Segmentation image is also acceptable if the background is labeled as 0 and the brain is labeled with other integer values than 0.')
    parser.add_argument('-m', '--mni152', type=str, required=True, help='path to the NIFTI file of the MNI152 template image, which is provided in data/template/MNI_152.nii.gz.')
    parser.add_argument('-a', '--age', type=float, required=False, help='(optional, but recommended) A float representing chronological age in years. This is used to correct a common bias in brain age estimation, where older subjects tend to be underestimated and younger subjects overestimated.')
    parser.add_argument('-s', '--sex', type=int, required=False, help='(optional, but recommended) integer, 0 for female, 1 for male.')
    parser.add_argument('-r', '--race', type=int, required=False, help='(optional, but recommended) integer, 1 for "White", 2 for "Asian", 3 for "Black or African American", 4 for "American Indian or Alaska Native", and 0 for "Some Other Race".')
    parser.add_argument('-w', '--weights', type=str, required=True, help='path to the directory of model weights pulled from Hugging Face. For example, /home-local/inference/braid-v1.0.')
    parser.add_argument('-i', '--intermediate', action='store_true', help='if the flag is given, the intermediate processing files will be preserved after the job completion. It is recommended to check the intermediate files, especially the processed images, before analysis.')
    parser.add_argument('-po', '--preprocess_only', action='store_true', help='if the flag is given, it will run everything except for the ResNet part. This is useful when the user want to run the preprocessing in parallel on a CPU-only machine (which may have more cores) first, and then switch to a GPU machine to complete the inference.')
    parser.add_argument('-o', '--outdir', type=str, required=True, help='path to the output directory (write permission required).')
    args = parser.parse_args()

    print("Start BRAID inference for one sample...")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # step 1: brain mask and skull-stripping for T1w image
    path_t1w_brain_mask = outdir / 't1w_brain_mask.nii.gz'
    if path_t1w_brain_mask.is_file():
        print(f'Brain mask for T1w image already exists at {path_t1w_brain_mask}')
    else:
        convert_to_brain_mask(args.t1w_mask, path_t1w_brain_mask)
        print(f'Saved brain mask for T1w image at {path_t1w_brain_mask}')

    path_t1w_brain = outdir / 't1w_brain.nii.gz'
    if path_t1w_brain.is_file():
        print(f'Skull-stripped T1w image already exists at {path_t1w_brain}')
    else:
        apply_skull_strip_mask(args.t1w, path_t1w_brain_mask, path_t1w_brain)
        print(f'Saved brain image at {path_t1w_brain}')

    # step 2: extract single shell and b0
    path_dwi_out = outdir / 'dwmri_firstshell.nii.gz'
    path_bval_out = outdir / 'dwmri_firstshell.bval'
    path_bvec_out = outdir / 'dwmri_firstshell.bvec'
    path_b0_out = outdir / 'dwmri_firstshell_b0.nii.gz'
    if path_dwi_out.is_file() and path_bval_out.is_file() and path_bvec_out.is_file() and path_b0_out.is_file():
        print(f'Single shell and b0 images already exist at {path_dwi_out.parent}')
    else:
        braid.calculate_dti_scalars.extract_single_shell(args.dwi, args.bval, args.bvec, path_dwi_out, path_bval_out, path_bvec_out, threshold=1500)
        braid.calculate_dti_scalars.extract_b0_volume(path_dwi_out, path_bval_out, path_bvec_out, path_b0_out)
        print(f'Extracted single shell and b0 images at {path_dwi_out.parent}')

    # step 3: registrations (b0 to t1, t1 to template affine, t1 to template non-rigid)
    transform_b0_to_t1 = outdir / 'transform_b0tot1.txt'
    transform_t1_to_b0 = outdir / 'transform_t1tob0.txt'
    transform_t1_to_template_affine = outdir / 'transform_t1toMNI_affine.mat'
    transform_t1_to_template_warp = outdir / 'transform_t1toMNI_warp.nii.gz'
    if transform_b0_to_t1.is_file() and transform_t1_to_b0.is_file() and transform_t1_to_template_affine.is_file() and transform_t1_to_template_warp.is_file():
        print(f'Registration files already exist at {transform_b0_to_t1.parent}')
    else:
        braid.registrations.register_b0_to_MNI152(path_b0_out, args.t1w, path_t1w_brain, args.mni152, outdir)
        print(f'Registration files saved at {transform_b0_to_t1.parent}')

    # step 4: calculate FA and MD maps
    path_b0_brain_mask = outdir / 'brain_mask_b0.nii.gz'
    path_b0_brain_mask_dilated = outdir / 'brain_mask_b0_dilated.nii.gz'
    path_tensor = outdir / 'tensor.nii.gz'
    path_fa = outdir / 'fa.nii.gz'
    path_md = outdir / 'md.nii.gz'
    if path_fa.is_file() and path_md.is_file() and path_tensor.is_file() and path_b0_brain_mask.is_file() and path_b0_brain_mask_dilated.is_file():
        print(f'FA and MD maps already exist at {path_fa.parent}')
    else:
        braid.calculate_dti_scalars.calculate_fa_md_maps(path_dwi_out, path_bval_out, path_bvec_out, path_b0_out, path_t1w_brain_mask, transform_t1_to_b0, path_b0_brain_mask, path_b0_brain_mask_dilated, path_tensor, path_fa, path_md)
        print(f'FA and MD maps saved at {path_fa.parent}')
    
    # step 5: skull-strip FA and MD maps
    path_fa_ss = outdir / 'fa_skullstrip.nii.gz'
    path_md_ss = outdir / 'md_skullstrip.nii.gz'

    if path_fa_ss.is_file() and path_md_ss.is_file():
        print(f'Skull-stripped FA and MD maps already exist at {path_fa_ss.parent}')
    else:
        apply_skull_strip_mask(path_fa, path_b0_brain_mask, path_fa_ss)
        apply_skull_strip_mask(path_md, path_b0_brain_mask, path_md_ss)
        print(f'Skull-stripped FA and MD maps saved at {path_fa_ss.parent}')

    # step 6: transform FA and MD maps to MNI152 space
    path_fa_ss_mni_affine = outdir / 'fa_skullstrip_MNI152.nii.gz'
    path_md_ss_mni_affine = outdir / 'md_skullstrip_MNI152.nii.gz'
    path_fa_ss_mni_warp = outdir / 'fa_skullstrip_MNI152_warped.nii.gz'
    path_md_ss_mni_warp = outdir / 'md_skullstrip_MNI152_warped.nii.gz'
    if path_fa_ss_mni_affine.is_file() and path_md_ss_mni_affine.is_file() and path_fa_ss_mni_warp.is_file() and path_md_ss_mni_warp.is_file():
        print(f'Transformed FA and MD maps already exist at {path_fa_ss_mni_affine.parent}')
    else:
        t_affine = [transform_t1_to_template_affine, transform_b0_to_t1]
        braid.registrations.apply_transform_to_img_in_b0(path_fa_ss, args.mni152, path_fa_ss_mni_affine, t_affine)
        braid.registrations.apply_transform_to_img_in_b0(path_md_ss, args.mni152, path_md_ss_mni_affine, t_affine)
        
        t_warp = [transform_t1_to_template_warp, transform_t1_to_template_affine, transform_b0_to_t1]
        braid.registrations.apply_transform_to_img_in_b0(path_fa_ss, args.mni152, path_fa_ss_mni_warp, t_warp)
        braid.registrations.apply_transform_to_img_in_b0(path_md_ss, args.mni152, path_md_ss_mni_warp, t_warp)