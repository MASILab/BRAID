""" Combine all datasets in the databank and collect path information along with demographic information.
The following path information will be collected:
    <databank_dti>
        - WMAtlas registration: so we can reuse the transformation matrices and the brain mask
        - PreQual: in case we need to do the registration, we can start from PreQualed data

    <databank_t1w>
        - T1w image: for future comparison between WM and GM age prediction
        - SLANT(_TICV) segmentation: for future skull stripping purpose

Version: 2.0
Author: Chenyu Gao
Date: Nov 14, 2023
"""

import re
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

# Columns of information to collect
list_df_dataset = [[],[]]
list_df_subject = [[],[]]
list_df_session = [[],[]]
list_df_scan = [[],[]]
list_df_sex = [[],[]]
list_df_race_simple = [[],[]]
list_df_race_original = [[],[]]
list_df_age = [[],[]]
list_df_diagnosis_simple = [[],[]]
list_df_diagnosis_original = [[],[]]
list_df_diagnosis_detail = [[],[]]
list_df_control_label = [[],[]]  # 1 for control, 0 for patient
list_df_path_image = [[],[]]  # Path to PreQual or T1w   (image)
list_df_path_deriv = [[],[]]  # Path to WMAtlas or SLANT (derivatives)

def save_to_lists(dataset_name, subject, session, scan, sex, race_simple, race_original, 
                  age, dx_simple, dx, dx_detail, control_label, 
                  path_image, path_deriv, modality='DTI'):
    global list_df_dataset, list_df_subject, list_df_session, list_df_scan, list_df_sex
    global list_df_race_simple, list_df_race_original, list_df_age
    global list_df_diagnosis_simple, list_df_diagnosis_original, list_df_diagnosis_detail
    global list_df_control_label, list_df_path_image, list_df_path_deriv
    
    if modality == 'DTI':
        idx_list = 0
    elif modality == 'T1w':
        idx_list = 1
    else:
        raise ValueError("Invalid modality. Supported modalities are 'DTI' and 'T1w'.")

    list_df_dataset[idx_list].append(dataset_name)
    list_df_subject[idx_list].append(subject)
    list_df_session[idx_list].append(session)
    list_df_scan[idx_list].append(scan)
    list_df_sex[idx_list].append(sex)
    list_df_race_simple[idx_list].append(race_simple)
    list_df_race_original[idx_list].append(race_original)
    list_df_age[idx_list].append(age)
    list_df_diagnosis_simple[idx_list].append(dx_simple)
    list_df_diagnosis_original[idx_list].append(dx)
    list_df_diagnosis_detail[idx_list].append(dx_detail)
    list_df_control_label[idx_list].append(control_label)
    list_df_path_image[idx_list].append(path_image)
    list_df_path_deriv[idx_list].append(path_deriv)

# Dictionary to remap diagnosis and race, lookup dataset location
dict_dx2standard = {
    'normal': 'normal',
    'Patient': 'patient',
    'LMCI': 'MCI',
    'impaired (not MCI)': 'patient',
    'No Cognitive Impairment': 'normal',
    'dementia': 'dementia',
    "Alzheimer's Dementia": 'dementia',
    'CN': 'normal',
    'MCI': 'MCI',
    'EMCI': 'MCI',
    'AD': 'dementia',
    'Mild Cognitive Impairment': 'MCI',
    'SMC': 'patient',
}

dict_race2standard = {
    'Native Hawaiian or Other Pacific Islander': 'Native Hawaiian or Other Pacific Islander',
    'Black or Black British': 'Black or African American',
    'Mixed': 'Some Other Race',
    'Other, Unknown, or More than one race': 'Some Other Race',
    'American Indian or Alaska Native': 'American Indian or Alaska Native',
    'White': 'White',
    'More than one race': 'Some Other Race',
    'Asian or Asian British': 'Asian',
    'Black or African American': 'Black or African American',
    'Other ethnic group': 'Some Other Race',
    'Other': 'Some Other Race',
    'Black': 'Black or African American',
    'Hispanic or Latino': 'Some Other Race',
    'Asian': 'Asian'
}

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
    # Retrieve information for each session (which can contain multiple scans)
    for _, row in tqdm(demog.iterrows(), total=demog.shape[0], desc=f'collecting info from {dataset_name}'):
        
        # esential information
        subject = row['subject']
        session = row['session'] if ('session' in row.keys()) else None
        age = row['age']
        sex = row['sex']
        race = row['race']
        if race in dict_race2standard.keys():
            race_simple = dict_race2standard[race]
        else:
            race_simple = None
        
        # diagnosis
        dx = row['diagnosis'] if ('diagnosis' in row.keys()) else None
        dx_simple = dict_dx2standard[dx] if (dx in dict_dx2standard.keys()) else None
        dx_detail = row['diagnosis_detail'] if ('diagnosis_detail' in row.keys()) else None
        if dx is not None:
            control_label = 1 if (dx_simple == 'normal') else 0
        else:
            control_label = row['CNS_control_2'] if ('CNS_control_2' in row.keys()) else None
        
        # DTI
        derivatives_folder = (dataset_path / 'derivatives' / subject) if pd.isna(session) else (dataset_path / 'derivatives' / subject / session)
        if derivatives_folder.is_dir():
            list_prequal = [fd for fd in derivatives_folder.iterdir() if ('PreQual' in fd.name) and ('DTIdouble' not in fd.name) and (fd/'PREPROCESSED'/'dwmri.nii.gz').is_file()]
        else:
            list_prequal = []
        list_prequal = sorted(list_prequal)
        for i, prequal_folder in enumerate(list_prequal):
            scan = i + 1  
            # WMAtlas folder (if any)
            wmatlas_folder = prequal_folder.parent / prequal_folder.name.replace('PreQual', 'WMAtlas')
            fa = wmatlas_folder / 'dwmri%fa.nii.gz'
            md = wmatlas_folder / 'dwmri%md.nii.gz'

            if fa.is_file() and md.is_file():
                save_to_lists(
                    dataset_name, subject, session, scan, sex, race_simple, race, age, 
                    dx_simple, dx, dx_detail, control_label, prequal_folder, wmatlas_folder, modality='DTI'
                )        
            else:
                if dataset_name != 'HCPA': print(wmatlas_folder, ' does not exist.')
                save_to_lists(
                    dataset_name, subject, session, scan, sex, race_simple, race, age, 
                    dx_simple, dx, dx_detail, control_label, prequal_folder, None, modality='DTI'
                )
        # T1w
        anat_folder = (dataset_path / subject / 'anat') if pd.isna(session) else (dataset_path / subject / session / 'anat')
        if anat_folder.is_dir():
            list_t1w = [fn for fn in anat_folder.iterdir() if '_T1w.nii' in fn.name]
        else:
            list_t1w = []
            
        for i, t1w in enumerate(list_t1w):
            scan = i+1
            _, t1w_seg = get_slantticv_path(t1w, derivatives_folder)
            save_to_lists(
                dataset_name, subject, session, scan, sex, race_simple, race, age, 
                dx_simple, dx, dx_detail, control_label, t1w, t1w_seg, modality='T1w')

# Convert to dataframe
df_dti = pd.DataFrame({
    'dataset': list_df_dataset[0],
    'subject': list_df_subject[0],
    'session': list_df_session[0],
    'scan': list_df_scan[0],
    'sex': list_df_sex[0],
    'race_simple': list_df_race_simple[0],
    'race_original': list_df_race_original[0],
    'age': list_df_age[0],
    'diagnosis_simple': list_df_diagnosis_simple[0],
    'diagnosis_original': list_df_diagnosis_original[0],
    'diagnosis_detail': list_df_diagnosis_detail[0],
    'control_label': list_df_control_label[0],
    'prequal_folder': list_df_path_image[0],
    'wmatlas_folder': list_df_path_deriv[0],
})
df_t1w = pd.DataFrame({
    'dataset': list_df_dataset[1],
    'subject': list_df_subject[1],
    'session': list_df_session[1],
    'scan': list_df_scan[1],
    'sex': list_df_sex[1],
    'race_simple': list_df_race_simple[1],
    'race_original': list_df_race_original[1],
    'age': list_df_age[1],
    'diagnosis_simple': list_df_diagnosis_simple[1],
    'diagnosis_original': list_df_diagnosis_original[1],
    'diagnosis_detail': list_df_diagnosis_detail[1],
    'control_label': list_df_control_label[1],
    't1w': list_df_path_image[1],
    'brainseg': list_df_path_deriv[1]
})

# Save to csv. The outputs must be on GDPR since it contains UKBB rows.
df_dti.to_csv('/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv', index=False)
df_t1w.to_csv('/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_t1w_v2.csv', index=False)