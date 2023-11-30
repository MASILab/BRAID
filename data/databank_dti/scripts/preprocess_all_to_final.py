# After collecting what masi has processed and doing the necessary preprocessing for HCPA,
# we have 12 datasets that have all files required for the final preprocessing step.
# Now we skull-strip FA, MD images, and then transform them to MNI152 space.
# 
# Author: Chenyu Gao
# Date: Nov 30, 2023

import braid.calculate_dti_scalars
import braid.registrations
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

def single_job_for_preprocessing(tuple_job):
    """wrapper for a single job of preprocessing (skull stripping and transform to MNI152)

    Args:
        tuple_job (tuple): (path_fa, path_md, path_brain_mask_b0, path_transform_b0tot1, path_transform_t1toMNI_affine, path_fa_ss, path_md_ss, path_fa_ss_mni, path_md_ss_mni)
    """
    
    global path_MNI152
    
    path_fa, path_md, path_brain_mask_b0, path_transform_b0tot1, path_transform_t1toMNI_affine, path_fa_ss, path_md_ss, path_fa_ss_mni, path_md_ss_mni = tuple_job
    
    # skull strip
    braid.calculate_dti_scalars.apply_skull_strip_mask(path_fa, path_brain_mask_b0, path_fa_ss)
    braid.calculate_dti_scalars.apply_skull_strip_mask(path_md, path_brain_mask_b0, path_md_ss)
    
    # transform to MNI152
    braid.registrations.apply_transform_to_img_in_b0(path_fa_ss, path_MNI152, path_fa_ss_mni, path_transform_b0tot1, path_transform_t1toMNI_affine)
    braid.registrations.apply_transform_to_img_in_b0(path_md_ss, path_MNI152, path_md_ss_mni, path_transform_b0tot1, path_transform_t1toMNI_affine)
    

path_databank_root = Path('/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_dti')
path_MNI152 = '/nfs2/ForChenyu/MNI_152.nii.gz'

print("collecting inputs for all jobs for parallel processing...")
list_tuple_job = []

for dataset in path_databank_root.iterdir():
    
    for subject in dataset.iterdir():
        if not (subject.name.startswith('sub-') and subject.is_dir()):
            continue
        
        for session in subject.iterdir():
            if not (session.name.startswith('ses-') and session.is_dir()):
                continue
            
            for scan in session.iterdir():
                if not (scan.name.startswith('scan-') and scan.is_dir()):
                    continue
                
                # inputs
                path_fa = scan / 'dti_fitting' / 'fa.nii.gz'
                path_md = scan / 'dti_fitting' / 'md.nii.gz'
                path_brain_mask_b0 = scan / 'brain_mask' / 'brain_mask_b0.nii.gz'
                path_transform_b0tot1 = scan / 'transform' / 'transform_b0tot1.txt'
                path_transform_t1toMNI_affine = scan / 'transform' / 'transform_t1toMNI_affine.mat'
                INPUTS_COMPLETE = path_fa.is_file() and path_md.is_file() and path_brain_mask_b0.is_file() and path_transform_b0tot1.is_file() and path_transform_t1toMNI_affine.is_file()
                
                # intermediate outputs
                path_fa_ss = scan / 'dti_fitting' / 'fa_skullstrip.nii.gz'
                path_md_ss = scan / 'dti_fitting' / 'md_skullstrip.nii.gz'
                
                # final outputs
                path_fa_ss_mni = scan / 'final' / 'fa_skullstrip_MNI152.nii.gz'
                path_md_ss_mni = scan / 'final' / 'md_skullstrip_MNI152.nii.gz'
                OUTPUTS_COMPLETE = path_fa_ss_mni.is_file() and path_md_ss_mni.is_file()
                
                if INPUTS_COMPLETE and not OUTPUTS_COMPLETE:
                    list_tuple_job.append((path_fa, path_md, path_brain_mask_b0, path_transform_b0tot1, path_transform_t1toMNI_affine, path_fa_ss, path_md_ss, path_fa_ss_mni, path_md_ss_mni))

print("Number of jobs created: {}\nAssign them to parallel workers...".format(len(list_tuple_job)))
with Pool(processes=8) as pool:
    results = list(tqdm(pool.imap(single_job_for_preprocessing, list_tuple_job, chunksize=1), total=len(list_tuple_job)))