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

def skull_stripping(job_tuple):
    """wrapper for a single job of skull stripping

    Args:
        tuple_job (tuple): 
    """
    
    fa, md, brain_mask_b0, fa_ss, md_ss = job_tuple
    
    if Path(fa_ss).is_file() and Path(md_ss).is_file():
        return
    
    if Path(fa).is_file() and Path(md).is_file() and Path(brain_mask_b0).is_file():
        braid.calculate_dti_scalars.apply_skull_strip_mask(fa, brain_mask_b0, fa_ss)
        braid.calculate_dti_scalars.apply_skull_strip_mask(md, brain_mask_b0, md_ss)
    

def apply_transforms(job_tuple):
    
    global path_MNI152
    transform_b0tot1, transform_t1toMNI_affine, transform_t1toMNI_warp, fa_ss, md_ss = job_tuple
    
    # output directory: scan-*/final
    outdir = Path(str(Path(fa_ss).parent).replace('dti_fitting', 'final'))
    
    fa_ss_mni_affine = outdir / 'fa_skullstrip_MNI152.nii.gz'
    md_ss_mni_affine = outdir / 'md_skullstrip_MNI152.nii.gz'
    fa_ss_mni_warp = outdir / 'fa_skullstrip_MNI152_warped.nii.gz'
    md_ss_mni_warp = outdir / 'md_skullstrip_MNI152_warped.nii.gz'    
    
    list_outputs = [fa_ss_mni_affine, md_ss_mni_affine, fa_ss_mni_warp, md_ss_mni_warp]
    if all([fn.is_file() for fn in list_outputs]):
        return
    
    list_inputs = [fa_ss, md_ss, transform_b0tot1, transform_t1toMNI_affine, transform_t1toMNI_warp]
    if not all([fn.is_file() for fn in list_inputs]):
        return
    
    # Affine transformed
    list_transforms = [transform_t1toMNI_affine, transform_b0tot1]
    braid.registrations.apply_transform_to_img_in_b0(fa_ss, path_MNI152, fa_ss_mni_affine, list_transforms)
    braid.registrations.apply_transform_to_img_in_b0(md_ss, path_MNI152, md_ss_mni_affine, list_transforms)
    
    # Non-linear warped
    list_transforms = [transform_t1toMNI_warp, transform_t1toMNI_affine, transform_b0tot1]
    braid.registrations.apply_transform_to_img_in_b0(fa_ss, path_MNI152, fa_ss_mni_warp, list_transforms)
    braid.registrations.apply_transform_to_img_in_b0(md_ss, path_MNI152, md_ss_mni_warp, list_transforms)


def generate_job_tuples(path_databank_root):
    dict_jobs = {"skull_stripping": [], "apply_transforms": []}
    
    for dataset in Path(path_databank_root).iterdir():
        print("\tsearching through dataset: {}".format(dataset.name))
    
        for subject in dataset.iterdir():
            if not (subject.name.startswith('sub-') and subject.is_dir()):
                continue
            
            for session in subject.iterdir():
                if not (session.name.startswith('ses-') and session.is_dir()):
                    continue
                
                for scan in session.iterdir():
                    if not (scan.name.startswith('scan-') and scan.is_dir()):
                        continue
                    
                    # input files
                    fa = scan / 'dti_fitting' / 'fa.nii.gz'
                    md = scan / 'dti_fitting' / 'md.nii.gz'
                    brain_mask_b0 = scan / 'brain_mask' / 'brain_mask_b0.nii.gz'
                    transform_b0tot1 = scan / 'transform' / 'transform_b0tot1.txt'
                    transform_t1toMNI_affine = scan / 'transform' / 'transform_t1toMNI_affine.mat'
                    transform_t1toMNI_warp = scan / 'transform' / 'transform_t1toMNI_warp.nii.gz'
                    
                    # intermediate outputs
                    fa_ss = scan / 'dti_fitting' / 'fa_skullstrip.nii.gz'
                    md_ss = scan / 'dti_fitting' / 'md_skullstrip.nii.gz'
                    
                    # add to job list
                    dict_jobs["skull_stripping"].append((
                        fa, md, brain_mask_b0, fa_ss, md_ss
                    ))
                    dict_jobs["apply_transforms"].append((
                        transform_b0tot1, transform_t1toMNI_affine, transform_t1toMNI_warp, fa_ss, md_ss
                    ))

    return dict_jobs


if __name__ == "__main__":
    path_MNI152 = '/nfs2/ForChenyu/MNI_152.nii.gz'
    
    dict_jobs = generate_job_tuples(path_databank_root='/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_dti')
    
    with Pool(processes=8) as pool:
        list(tqdm(pool.imap(skull_stripping, dict_jobs["skull_stripping"], chunksize=1), total=len(dict_jobs["skull_stripping"]), desc='Skull stripping'))
        list(tqdm(pool.imap(apply_transforms, dict_jobs["apply_transforms"], chunksize=1), total=len(dict_jobs["apply_transforms"]), desc='Applying transforms'))
        