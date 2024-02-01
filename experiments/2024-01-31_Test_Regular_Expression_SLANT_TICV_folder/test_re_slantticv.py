""" Use regular expression to find the SLANT-TICV folder and segmentation image.

Author: Michael E. Kim and Chenyu Gao
Date: Jan 31, 2024
"""
import re
import os
import pdb
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def get_slantticv_path(path_t1w, path_derivative):
    """ Find the path to the corresponding SLANT-TICV results for the t1w.
    credit to Michael E. Kim.
    """
    pattern = r'(sub-\w+)(?:_(ses-\w+))?(?:_(acq-\w+))?(?:_(run-\d{1,2}))?_T1w'
    matches = re.findall(pattern, path_t1w.name)
    sub, ses, acq, run = matches[0]
    
    # Find the TICV directory
    ticv_dir = path_derivative / f'SLANT-TICVv1.2{acq}{run}'
    if not ticv_dir.is_dir():
        ticv_dir = path_derivative / f"SLANT-TICVv1.2{acq}{run.replace('run-', 'run-0')}"
        if not ticv_dir.is_dir():
            list_ticv_dirs = [fd for fd in path_derivative.iterdir() if ('SLANT-TICVv1.2' in fd.name) and (fd.is_dir())]
            if len(list_ticv_dirs) == 1:
                ticv_dir = path_derivative / 'SLANT-TICVv1.2'
            else:
                return None, None

    # Find the segmentation file
    if not ses == '':
        ses = '_' + ses
    if not acq == '':
        acq = '_' + acq
    if not run == '':
        run = '_' + run
    
    possible_seg_paths = [
        (ticv_dir / 'out' / 'post' / 'FinalResult' / "{}{}{}{}_T1w_seg.nii.gz".format(sub, ses, acq, run)),
        (ticv_dir / 'post' / 'FinalResult' / "{}{}{}{}_T1w_seg.nii.gz".format(sub, ses, acq, run)),
        (ticv_dir / 'FinalResult' / "{}{}{}{}_T1w_seg.nii.gz".format(sub, ses, acq, run)),
        (ticv_dir / "{}{}{}{}_T1w_seg.nii.gz".format(sub, ses, acq, run)),
    ]
    img_seg = None
    for p in possible_seg_paths:
        if p.is_file():
            img_seg = p
            break

    return ticv_dir, img_seg

        
dict_bids_location = {
    'OASIS3': '/nfs2/harmonization/BIDS/OASIS3',
    'WRAP': '/nfs2/harmonization/BIDS/WRAP',
    'BIOCARD': '/nfs2/harmonization/BIDS/BIOCARD',
    'OASIS4': '/nfs2/harmonization/BIDS/OASIS4',
    'BLSA': '/nfs2/harmonization/BIDS/BLSA',
    'NACC': '/nfs2/harmonization/BIDS/NACC',
    'ROSMAPMARS': '/nfs2/harmonization/BIDS/ROSMAPMARS',
    'UKBB': '/home/gaoc11/GDPR/BIDS/UKBB',
    'VMAP': '/nfs2/harmonization/BIDS/VMAP',
    'ADNI': '/nfs2/harmonization/BIDS/ADNI_DTI',
    'ICBM': '/nfs2/harmonization/BIDS/ICBM',
    'HCPA': '/nfs2/harmonization/BIDS/HCPA',
}

# Folder of the cleaned csv
clean_csv_root = Path('/nfs/masi/gaoc11/projects/BRAID/data/subject_info/clean')
list_clean_csv = [fd for fd in clean_csv_root.iterdir() if fd.name.endswith('_info.csv')]
list_clean_csv.append(Path('/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/subject_info/clean/UKBB_info.csv'))  # UKBB is stored on GDPR

# Load csv for each dataset
for csv in list_clean_csv:

    demog = pd.read_csv(csv)
    
    dataset_name = csv.name.split('_')[0]
    dataset_path = Path(dict_bids_location[dataset_name])
    if dataset_name in ['OASIS3', 'WRAP', 'BIOCARD', 'OASIS4', 'BLSA', 'NACC', 'ROSMAPMARS', 'VMAP', 'ADNI', 'ICBM', 'UKBB']:
        print(f"{dataset_name} has been checked.")
        continue
            
    # Retrieve information for each session (which can contain multiple scans)
    for _,row in tqdm(demog.iterrows(), total=demog.shape[0], desc=f'searching through {dataset_name}'):
        
        subject = row['subject']
        session = row['session'] if ('session' in row.keys()) else None

        derivatives_folder = (dataset_path / 'derivatives' / subject) if pd.isna(session) else (dataset_path / 'derivatives' / subject / session)

        anat_folder = (dataset_path / subject / 'anat') if pd.isna(session) else (dataset_path / subject / session / 'anat')

        if anat_folder.is_dir():
            list_t1w = [fn for fn in anat_folder.iterdir() if '_T1w.nii' in fn.name]
        else:
            list_t1w = []
            
        for i, t1w in enumerate(list_t1w):
            ticv_dir, img_seg = get_slantticv_path(t1w, derivatives_folder)
            if img_seg is None:
                print(t1w)
