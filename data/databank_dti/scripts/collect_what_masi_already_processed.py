# Set up the databank_dti in its initial state, with the data that MASI has already processed.
# Store everything on GDPR to comply with the regulations.
# Run this script on hickory.
# 
# Author: Chenyu Gao
# Date: Nov 20, 2023

import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Location of the databank_dti
path_databank_root = Path('/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_dti')  # on hickory

# Load the csv of imaging paths of all datasets
df = pd.read_csv('/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti.csv')
df = df.loc[df['age'].notnull(), ]

for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    
    dataset = row['dataset']
    subject = row['subject']
    session = row['session'] if pd.notnull(row['session']) else 'ses-1'
    scan = 'scan-{}'.format(row['scan'])

    # If WMAtlas exists, then we have all files required to preprocessing
    if pd.notnull(row['wmatlas_folder']):
        
        # DTI Scalar maps
        path_fa_img = Path(row['wmatlas_folder']) / 'dwmri%fa.nii.gz'
        path_md_img = Path(row['wmatlas_folder']) / 'dwmri%md.nii.gz'
        # Brain mask in B0 space
        path_b0_mask_img = Path(row['wmatlas_folder']) / 'dwmri%_dwimask.nii.gz'
        # Transformations
        path_transform_b0_to_t1 = Path(row['wmatlas_folder']) / 'dwmri%ANTS_b0tot1.txt'
        path_transform_t1_to_MNI_affine = Path(row['wmatlas_folder']) / 'dwmri%0GenericAffine.mat'

        # Check their existence
        list_files_to_check = [path_fa_img, path_md_img, path_b0_mask_img, path_transform_b0_to_t1, path_transform_t1_to_MNI_affine]
        if any(not file.exists() for file in list_files_to_check):
            print(row['wmatlas_folder'], ' has missing files.')
            continue

        # Save the above to the databank_dti
        t_path_dti_fitting_folder = path_databank_root / dataset / subject / session / scan / 'dti_fitting'
        t_path_brain_mask_folder = path_databank_root / dataset / subject / session / scan / 'brain_mask'
        t_path_transform_folder = path_databank_root / dataset / subject / session / scan / 'transform'
        t_path_final_folder = path_databank_root / dataset / subject / session / scan / 'final'

        subprocess.run(['mkdir', '-p', str(t_path_dti_fitting_folder)])
        subprocess.run(['mkdir', '-p', str(t_path_brain_mask_folder)])
        subprocess.run(['mkdir', '-p', str(t_path_transform_folder)])
        subprocess.run(['mkdir', '-p', str(t_path_final_folder)])

        t_path_fa_img = t_path_dti_fitting_folder / 'fa.nii.gz'
        t_path_md_img = t_path_dti_fitting_folder / 'md.nii.gz'
        t_path_b0_mask_img = t_path_brain_mask_folder / 'brain_mask_b0.nii.gz'
        t_path_transform_b0_to_t1 = t_path_transform_folder / 'transform_b0tot1.txt'
        t_path_transform_t1_to_MNI_affine = t_path_transform_folder / 'transform_t1toMNI_affine.mat'

        # subprocess.run(['rsync', '-L', str(path_fa_img), str(t_path_fa_img)])
        # subprocess.run(['rsync', '-L', str(path_md_img), str(t_path_md_img)])
        # subprocess.run(['rsync', '-L', str(path_b0_mask_img), str(t_path_b0_mask_img)])
        # subprocess.run(['rsync', '-L', str(path_transform_b0_to_t1), str(t_path_transform_b0_to_t1)])
        # subprocess.run(['rsync', '-L', str(path_transform_t1_to_MNI_affine), str(t_path_transform_t1_to_MNI_affine)])
        subprocess.run(['ln', '-s', str(path_fa_img), str(t_path_fa_img)])
        subprocess.run(['ln', '-s', str(path_md_img), str(t_path_md_img)])
        subprocess.run(['ln', '-s', str(path_b0_mask_img), str(t_path_b0_mask_img)])
        subprocess.run(['ln', '-s', str(path_transform_b0_to_t1), str(t_path_transform_b0_to_t1)])
        subprocess.run(['ln', '-s', str(path_transform_t1_to_MNI_affine), str(t_path_transform_t1_to_MNI_affine)])
    

    # If WMAtlas does not exist but it's HCPA, then we can preprocess it
    elif dataset == 'HCPA':

        # prequal
        path_prequal_bval = Path(row['prequal_folder']) / 'PREPROCESSED' / 'dwmri.bval'
        path_prequal_bvec = Path(row['prequal_folder']) / 'PREPROCESSED' / 'dwmri.bvec'
        path_prequal_dmri = Path(row['prequal_folder']) / 'PREPROCESSED' / 'dwmri.nii.gz'
        # t1w
        path_anat_folder = Path(row['prequal_folder'].replace('derivatives/', '').replace(row['prequal_folder'].split('/')[-1], 'anat'))
        list_t1w = [fn for fn in path_anat_folder.iterdir() if '_T1w.nii' in fn.name]
        if len(list_t1w) == 0:
            continue
        list_t1w = sorted(list_t1w)
        path_t1w_img = list_t1w[0]

        # Check their existence
        list_files_to_check = [path_prequal_bval, path_prequal_bvec, path_prequal_dmri]
        if any(not file.exists() for file in list_files_to_check):
            print(row['prequal_folder'], ' lacking enough files for preproc.')
            continue

        # Save the above to the databank_dti
        t_path_prequal_folder = path_databank_root / dataset / subject / session / scan / 'prequal'
        t_path_t1w_folder = path_databank_root / dataset / subject / session / scan / 't1w'
        subprocess.run(['mkdir', '-p', str(t_path_prequal_folder)])
        subprocess.run(['mkdir', '-p', str(t_path_t1w_folder)])

        t_path_prequal_bval = t_path_prequal_folder / 'dwmri.bval'
        t_path_prequal_bvec = t_path_prequal_folder / 'dwmri.bvec'
        t_path_prequal_dmri = t_path_prequal_folder / 'dwmri.nii.gz'
        t_path_t1w_img = t_path_t1w_folder / 't1w.nii.gz'

        # subprocess.run(['rsync', '-L', str(path_prequal_bval), str(t_path_prequal_bval)])
        # subprocess.run(['rsync', '-L', str(path_prequal_bvec), str(t_path_prequal_bvec)])
        # subprocess.run(['rsync', '-L', str(path_prequal_dmri), str(t_path_prequal_dmri)])
        # subprocess.run(['rsync', '-L', str(path_t1w_img), str(t_path_t1w_img)])
        subprocess.run(['ln', '-s', str(path_prequal_bval), str(t_path_prequal_bval)])
        subprocess.run(['ln', '-s', str(path_prequal_bvec), str(t_path_prequal_bvec)])
        subprocess.run(['ln', '-s', str(path_prequal_dmri), str(t_path_prequal_dmri)])
        subprocess.run(['ln', '-s', str(path_t1w_img), str(t_path_t1w_img)])


    else:
        continue
