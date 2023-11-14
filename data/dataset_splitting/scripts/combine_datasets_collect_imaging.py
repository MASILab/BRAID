# Should run this script on hickory to access GDPR (UKBB).
# Combine all datasets in the data bank and collect path information along with demographic information.
# 
# Author: Chenyu Gao
# Date: Nov 14, 2023

import pandas as pd
from pathlib import Path
from tqdm import tqdm

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
list_df_t1w_path = []  # Path to T1w image (not necessarily the one used for PreQual). We will probably need it for comparison between WM and GM age prediction.
list_df_dti_registration_folder = []

def save_to_lists(dataset_name, subject, session, scan, sex, race_simple, race_original, age, dx_simple, dx, dx_detail, control_label, path, modality='DTI'):
    global list_df_dataset, list_df_subject, list_df_session, list_df_scan, list_df_sex
    global list_df_race_simple, list_df_race_original, list_df_age
    global list_df_diagnosis_simple, list_df_diagnosis_original, list_df_diagnosis_detail
    global list_df_control_label, list_df_t1w_path, list_df_dti_registration_folder
    
    if modality == 'DTI':
        list_df_dataset[0].append(dataset_name)
        list_df_subject[0].append(subject)
        list_df_session[0].append(session)
        list_df_scan[0].append(scan)
        list_df_sex[0].append(sex)
        list_df_race_simple[0].append(race_simple)
        list_df_race_original[0].append(race_original)
        list_df_age[0].append(age)
        list_df_diagnosis_simple[0].append(dx_simple)
        list_df_diagnosis_original[0].append(dx)
        list_df_diagnosis_detail[0].append(dx_detail)
        list_df_control_label[0].append(control_label)
        list_df_dti_registration_folder.append(path)
    
    elif modality == 'T1w':
        list_df_dataset[1].append(dataset_name)
        list_df_subject[1].append(subject)
        list_df_session[1].append(session)
        list_df_scan[1].append(scan)
        list_df_sex[1].append(sex)
        list_df_race_simple[1].append(race_simple)
        list_df_race_original[1].append(race_original)
        list_df_age[1].append(age)
        list_df_diagnosis_simple[1].append(dx_simple)
        list_df_diagnosis_original[1].append(dx)
        list_df_diagnosis_detail[1].append(dx_detail)
        list_df_control_label[1].append(control_label)
        list_df_t1w_path.append(path)
    
    else:
        raise ValueError("Invalid modality. Supported modalities are 'DTI' and 'T1w'.")
    
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
}

# Folder of the cleaned csv
clean_csv_root = Path('./data/subject_info/clean')

# Load csv for each dataset
for csv in clean_csv_root.iterdir():
    if csv.suffix != '.csv':
        continue
    demog = pd.read_csv(csv)
    
    dataset_name = csv.name.split('_')[0]
    print('Start processing dataset: ', dataset_name)
    dataset_path = Path(dict_bids_location[dataset_name])
            
    # Retrieve information for each session (which can contain multiple scans)
    for _,row in tqdm(demog.iterrows(), total=demog.shape[0]):
        
        # esential information that every dataset should have
        subject = row['subject']
        race = row['race']
        if race in dict_race2standard.keys():
            race_simple = dict_race2standard[race]
        else:
            race_simple = None
        sex = row['sex']
        
        # dataset-specific information
        if dataset_name == 'OASIS3':
            session = row['session']
            age = row['age']
            dx = row['diagnosis']
            if dx in dict_dx2standard.keys():
                dx_simple = dict_dx2standard[dx]
            else:
                dx_simple = None
            dx_detail = None
                
            if dx_simple == 'normal':
                control_label = 1
            else:
                control_label = 0
            
            # DTI
            derivatives_folder = dataset_path / 'derivatives' / subject / session
            list_wmatlas = [fd for fd in derivatives_folder.iterdir() if ('WMAtlas' in fd.name) and ('EVE3' not in fd.name)]
            list_wmatlas = sorted(list_wmatlas)
            for i, dti_registration_folder in enumerate(list_wmatlas):
                # check if it is empty
                fa = dti_registration_folder / 'dwmri%fa.nii.gz'
                md = dti_registration_folder / 'dwmri%md.nii.gz'
                if fa.is_file() and md.is_file():
                    scan = i+1
                    save_to_lists(dataset_name, subject, session, scan, sex, race_simple, race, age, dx_simple, dx, dx_detail, control_label, dti_registration_folder, modality='DTI')

            # T1w
            anat_folder = dataset_path / subject / session / 'anat'
            if anat_folder.is_dir():
                list_t1w = [fn for fn in anat_folder.iterdir() if '_T1w.nii' in fn.name]
            else:
                list_t1w = []

            for i, t1w in enumerate(list_t1w):
                scan = i+1
                save_to_lists(dataset_name, subject, session, scan, sex, race_simple, race, age, dx_simple, dx, dx_detail, control_label, t1w, modality='T1w')


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
    'dti_registration_folder': list_df_dti_registration_folder,
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
    't1w_nifti': list_df_t1w_path,
})

# Save to csv
df_dti.to_csv('./data/dataset_splitting/spreadsheet/databank_dti.csv', index=False)
df_t1w.to_csv('./data/dataset_splitting/spreadsheet/databank_t1w.csv', index=False)