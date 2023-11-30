import subprocess
import braid.calculate_dti_scalars as cds
from pathlib import Path
from tqdm import tqdm

path_dataset = Path('/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_dti/HCPA')
list_subject = [subject for subject in path_dataset.iterdir() if subject.name.startswith('sub-') and subject.is_dir()]

# # Skull stripping
# print('Start skull stripping...')
# list_todo_t1w = []
# list_todo_t1w_brain = []
# list_todo_t1w_brain_mask = []

# for subject in path_dataset.iterdir():
#     if not (subject.name.startswith('sub-') and subject.is_dir()):
#         continue
    
#     for session in subject.iterdir():
#         if not (session.name.startswith('ses-') and session.is_dir()):
#             continue
        
#         for scan in session.iterdir():
#             if not (scan.name.startswith('scan-') and scan.is_dir()):
#                 continue
            
#             path_t1w = scan / 't1w' / 't1w.nii.gz'
#             path_t1w_brain = scan / 't1w_brain_mask' / 't1w_brain.nii.gz'
#             path_t1w_brain_mask = scan / 't1w_brain_mask' / 't1w_brain_mask.nii.gz'
            
#             if path_t1w.is_file() and not (path_t1w_brain.is_file() and path_t1w_brain_mask.is_file()):
#                 list_todo_t1w.append(path_t1w)
#                 list_todo_t1w_brain.append(path_t1w_brain)
#                 list_todo_t1w_brain_mask.append(path_t1w_brain_mask)
                
# for path_t1w, path_t1w_brain, path_t1w_brain_mask in tqdm(zip(list_todo_t1w, list_todo_t1w_brain, list_todo_t1w_brain_mask), total=len(list_todo_t1w)):
#     subprocess.run(['mkdir', '-p', str(path_t1w_brain.parent)])            
#     cds.run_mri_synthstrip(path_t1w, path_t1w_brain, path_t1w_brain_mask)

# print('Skull stripping done.')

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
                cds.extract_single_shell(path_dwi_in, path_bval_in, path_bvec_in, path_dwi_out, path_bval_out, path_bvec_out, threshold=1500)
                cds.extract_b0_volume(path_dwi_out, path_bval_out, path_bvec_out, path_b0_out)
            
            print('Extracted single shell and b0 for {}'.format(path_dwi_out.parent))
                            
print('Extracting single shell and b0 done.')

# Registrations (b0 to MNI152): prepare transformation files
print('Start registrations...')

for subject in tqdm(path_dataset.iterdir(), total=len(list_subject)):
    if not (subject.name.startswith('sub-') and subject.is_dir()):
        continue
    
    for session in subject.iterdir():
        if not (session.name.startswith('ses-') and session.is_dir()):
            continue
        
        for scan in session.iterdir():
            if not (scan.name.startswith('scan-') and scan.is_dir()):
                continue

