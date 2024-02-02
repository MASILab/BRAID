""" Prepare required preprocessing files that other datasets already have for HCPA dataset.
"""

import subprocess
import braid.calculate_dti_scalars
import braid.registrations
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

def single_job_for_parallel_register_b0_to_MNI152(tuple_job):
    """wrapper for parallel processing of register_b0_to_MNI152()

    Args:
        tuple_job (tuple): (path_b0, path_t1, path_t1_brain, path_MNI152, outdir)
    """
    path_b0, path_t1, path_t1_brain, path_MNI152, outdir = tuple_job
    braid.registrations.register_b0_to_MNI152(path_b0, path_t1, path_t1_brain, path_MNI152, outdir)


path_dataset = Path('/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_dti/HCPA')
list_subject = [subject for subject in path_dataset.iterdir() if subject.name.startswith('sub-') and subject.is_dir()]

# Skull stripping
print('Start skull stripping...')
list_todo_t1w = []
list_todo_t1w_brain = []
list_todo_t1w_brain_mask = []

for subject in path_dataset.iterdir():
    if not (subject.name.startswith('sub-') and subject.is_dir()):
        continue
    
    for session in subject.iterdir():
        if not (session.name.startswith('ses-') and session.is_dir()):
            continue
        
        for scan in session.iterdir():
            if not (scan.name.startswith('scan-') and scan.is_dir()):
                continue
            
            path_t1w = scan / 't1w' / 't1w.nii.gz'
            path_t1w_brain = scan / 't1w_brain_mask' / 't1w_brain.nii.gz'
            path_t1w_brain_mask = scan / 't1w_brain_mask' / 't1w_brain_mask.nii.gz'
            
            if path_t1w.is_file() and not (path_t1w_brain.is_file() and path_t1w_brain_mask.is_file()):
                list_todo_t1w.append(path_t1w)
                list_todo_t1w_brain.append(path_t1w_brain)
                list_todo_t1w_brain_mask.append(path_t1w_brain_mask)
                
for path_t1w, path_t1w_brain, path_t1w_brain_mask in tqdm(zip(list_todo_t1w, list_todo_t1w_brain, list_todo_t1w_brain_mask), total=len(list_todo_t1w)):
    subprocess.run(['mkdir', '-p', str(path_t1w_brain.parent)])            
    braid.calculate_dti_scalars.run_mri_synthstrip(path_t1w, path_t1w_brain, path_t1w_brain_mask)

print('Skull stripping done.')

# Extract single shell and b0
print('Start extracting single shell and b0...')

for subject in tqdm(path_dataset.iterdir(), total=len(list_subject)):
    if not (subject.name.startswith('sub-') and subject.is_dir()):
        continue
    
    for session in subject.iterdir():
        if not (session.name.startswith('ses-') and session.is_dir()):
            continue
        
        for scan in session.iterdir():
            if not (scan.name.startswith('scan-') and scan.is_dir()):
                continue
            
            path_dwi_in = scan / 'prequal' / 'dwmri.nii.gz'
            path_bval_in = scan / 'prequal' / 'dwmri.bval'
            path_bvec_in = scan / 'prequal' / 'dwmri.bvec'
            
            path_dwi_out = scan / 'firstshell' / 'dwmri_firstshell.nii.gz'
            path_bval_out = scan / 'firstshell' / 'dwmri_firstshell.bval'
            path_bvec_out = scan / 'firstshell' / 'dwmri_firstshell.bvec'
            
            path_b0_out = scan / 'firstshell' / 'dwmri_firstshell_b0.nii.gz'
            
            INPUT_COMPLETE = path_dwi_in.is_file() and path_bval_in.is_file() and path_bvec_in.is_file()
            OUTPUT_COMPLETE = path_dwi_out.is_file() and path_bval_out.is_file() and path_bvec_out.is_file() and path_b0_out.is_file()
            
            if INPUT_COMPLETE and not OUTPUT_COMPLETE:
                subprocess.run(['mkdir', '-p', str(path_dwi_out.parent)])
                braid.calculate_dti_scalars.extract_single_shell(path_dwi_in, path_bval_in, path_bvec_in, path_dwi_out, path_bval_out, path_bvec_out, threshold=1500)
                braid.calculate_dti_scalars.extract_b0_volume(path_dwi_out, path_bval_out, path_bvec_out, path_b0_out)
            
            print('Extracted single shell and b0 for {}'.format(path_dwi_out.parent))
                            
print('Extracting single shell and b0 done.')

# Registrations (b0 to MNI152): prepare transformation files
print('Preparing list of jobs for registrations in parallel...')
path_MNI152 = '/nfs2/ForChenyu/MNI_152.nii.gz'

list_tuple_jobs = []

for subject in tqdm(path_dataset.iterdir(), total=len(list_subject)):
    if not (subject.name.startswith('sub-') and subject.is_dir()):
        continue
    
    for session in subject.iterdir():
        if not (session.name.startswith('ses-') and session.is_dir()):
            continue
        
        for scan in session.iterdir():
            if not (scan.name.startswith('scan-') and scan.is_dir()):
                continue
            
            # inputs
            path_b0 = scan / 'firstshell' / 'dwmri_firstshell_b0.nii.gz'
            path_t1 = scan / 't1w' / 't1w.nii.gz'
            path_t1_brain = scan / 't1w_brain_mask' / 't1w_brain.nii.gz'
            INPUTS_COMPLETE = path_b0.is_file() and path_t1.is_file() and path_t1_brain.is_file()
            
            # outputs
            outdir = scan / 'transform'
            b0_to_t1_ants = outdir / 'transform_b0tot1.txt'
            t1_to_b0_ants = outdir / 'transform_t1tob0.txt'
            t1_to_template_affine = outdir / 'transform_t1toMNI_affine.mat'
            OUTPUTS_COMPLETE = b0_to_t1_ants.is_file() and t1_to_b0_ants.is_file() and t1_to_template_affine.is_file()

            if INPUTS_COMPLETE and not OUTPUTS_COMPLETE:
                list_tuple_jobs.append((path_b0, path_t1, path_t1_brain, path_MNI152, outdir))

print('Number of jobs created: {}\nAssign them to parallel workers...'.format(len(list_tuple_jobs)))
with Pool(processes=8) as pool:
    results = list(tqdm(pool.imap(single_job_for_parallel_register_b0_to_MNI152, list_tuple_jobs, chunksize=1), total=len(list_tuple_jobs)))
print('Registrations done.')


# Calculate the FA and MD maps (will need conda environment utls)
print('Start calculating FA and MD maps...')
for subject in tqdm(path_dataset.iterdir(), total=len(list_subject)):
    if not (subject.name.startswith('sub-') and subject.is_dir()):
        continue
    
    for session in subject.iterdir():
        if not (session.name.startswith('ses-') and session.is_dir()):
            continue
        
        for scan in session.iterdir():
            if not (scan.name.startswith('scan-') and scan.is_dir()):
                continue
            
            # inputs
            path_dwi = scan / 'firstshell' / 'dwmri_firstshell.nii.gz'
            path_bval = scan / 'firstshell' / 'dwmri_firstshell.bval'
            path_bvec = scan / 'firstshell' / 'dwmri_firstshell.bvec'
            path_b0 = scan / 'firstshell' / 'dwmri_firstshell_b0.nii.gz'
            path_t1_brain_mask = scan / 't1w_brain_mask' / 't1w_brain_mask.nii.gz'
            path_transform_t1tob0 = scan / 'transform' / 'transform_t1tob0.txt'
            INPUTS_COMPLETE = path_dwi.is_file() and path_bval.is_file() and path_bvec.is_file() and path_b0.is_file() and path_t1_brain_mask.is_file() and path_transform_t1tob0.is_file()
            
            # outputs
            path_b0_brain_mask = scan / 'brain_mask' / 'brain_mask_b0.nii.gz'
            path_b0_brain_mask_dilated = scan / 'brain_mask' / 'brain_mask_b0_dilated.nii.gz'
            path_tensor = scan / 'dti_fitting' / 'tensor.nii.gz'
            path_fa = scan / 'dti_fitting' / 'fa.nii.gz'
            path_md = scan / 'dti_fitting' / 'md.nii.gz'
            OUTPUTS_COMPLETE = path_b0_brain_mask.is_file() and path_b0_brain_mask_dilated.is_file() and path_tensor.is_file() and path_fa.is_file() and path_md.is_file()     
            
            if INPUTS_COMPLETE and not OUTPUTS_COMPLETE:
                braid.calculate_dti_scalars.calculate_fa_md_maps(path_dwi, path_bval, path_bvec, path_b0, path_t1_brain_mask, path_transform_t1tob0, path_b0_brain_mask, path_b0_brain_mask_dilated, path_tensor, path_fa, path_md)
print('Calculating FA and MD maps done.')
print('Now HCPA should contain all necessary files for completing preprocessing,\njust like the first 11 datasets!!!')
