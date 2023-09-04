# Collect information of interest.
# 
# Author: Chenyu Gao
# Date: Sept 4, 2023

import json
import pandas as pd
from pathlib import Path

path_dataset_bids = Path('/nfs2/harmonization/BIDS/OASIS4')

df_imaging = pd.read_csv('./data/subject_info/raw/OASIS4/imaging/csv/OASIS4_data_imaging.csv')  # age at imaging
df_clinical = pd.read_csv('./data/subject_info/raw/OASIS4/clinical/csv/OASIS4_data_clinical.csv')  # sex, race, diagnosis

dict_sex2standard = {
    1: 'male',
    2: 'female',
}
dict_race2standard = {
    1: 'White',
    2: 'Black or African American',
    3: 'American Indian or Alaska Native',
    4: 'Native Hawaiian or Other Pacific Islander',
    5: 'Asian',
    0: 'Other',
    99: None,
}
dict_dx2standard = {
    'Uncertain - AD possible': None,
    'FTD': None,
    'Alzheimer Disease Dementia': 'AD',
    'MCI': 'MCI',
    'DLB': 'dementia',
    'AD Variant': 'AD',
    'Non-Neurodegenerative Neurologic Disease': None,
    'Other Non-AD Neurodegenerative Disorder': None,
    'Early Onset AD': 'AD',
    'AD+Non Neurodegenerative': 'AD', 
    'AD/Vascular': 'AD',
    'Cognitively Normal': 'normal',
    'Other - Miscellaneous': None, 
    'Vascular Cognitive Impairment (VCI)': None,
    'Mood/polypharmacy/sleep': None, 
    'PPA': None,
}

list_df_subject = []
list_df_session = []
list_df_race = []
list_df_sex = []
list_df_age = []
list_df_diagnosis = []
list_df_diagnosis_detail = []

# Loop through sessions we have, and collect information
for subject in path_dataset_bids.iterdir():
    if not (subject.is_dir() and subject.name.startswith('sub-')):
        continue
    
    for session in subject.iterdir():
        if not (session.is_dir() and session.name.startswith('ses-')):
            continue
        
        imaging_id = f"{subject.name.split('sub-')[1]}_MR_{session.name.split('ses-')[1]}"
        oasis_id = subject.name.split('sub-')[1]
        
        try:
            values = df_clinical.loc[df_clinical['oasis_id']==oasis_id, 'race'].values
            values = values[~pd.isna(values)]
            race = values[0]
            race = dict_race2standard[race]
        except:
            race = None
        
        try:
            values = df_clinical.loc[df_clinical['oasis_id']==oasis_id, 'sex'].values
            values = values[~pd.isna(values)]
            sex = values[0]
            sex = dict_sex2standard[sex]
        except:
            sex = None
        
        try:
            values = df_clinical.loc[df_clinical['oasis_id']==oasis_id, 'final_dx'].values
            values = values[~pd.isna(values)]
            dx_detail = values[0]
        except:
            dx_detail = None
        
        try:
            dx = dict_dx2standard[dx_detail]
        except:
            dx = None
        
        try:
            values = df_imaging.loc[df_imaging['imaging_id']==imaging_id, 'image_age'].values
            values = values[~pd.isna(values)]
            age = values[0]
        except:
            age = None

        list_df_subject.append(subject.name)
        list_df_session.append(session.name)
        list_df_race.append(race)
        list_df_sex.append(sex)
        list_df_age.append(age)
        list_df_diagnosis.append(dx)
        list_df_diagnosis_detail.append(dx_detail)
        
d = {'subject': list_df_subject,
     'session': list_df_session,
     'race': list_df_race,
     'sex': list_df_sex,
     'age': list_df_age,
     'diagnosis': list_df_diagnosis,
     'diagnosis_detail': list_df_diagnosis_detail,
}
df = pd.DataFrame(data=d)
df.sort_values(by=['subject', 'age'], inplace=True)

df.to_csv('./data/subject_info/clean/OASIS4_info.csv', index=False)
