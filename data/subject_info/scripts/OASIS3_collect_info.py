# Collect information of interest.
# 
# Author: Chenyu Gao
# Date: Sept 4, 2023

import json
import pandas as pd
from pathlib import Path

path_dataset_bids = Path('/nfs2/harmonization/BIDS/OASIS3')

demog = pd.read_csv('./data/subject_info/raw/OASIS3/demo/csv/OASIS3_demographics.csv')
diagnosis = pd.read_csv('./data/subject_info/raw/OASIS3/UDSd1/csv/OASIS3_UDSd1_diagnoses.csv')

dict_sex2standard = {
    1: 'male',
    2: 'female',
}
dict_race2standard = {
    'White': 'White',
    'ASIAN': 'Asian',
    'Black': 'Black or African American',
    'more than one': 'Other',
    'Unknown': None,
    'AIAN': 'American Indian or Alaska Native',
}

list_df_subject = []
list_df_session = []
list_df_race = []
list_df_sex = []
list_df_age = []
list_df_diagnosis = []

# Loop through sessions we have, and collect information
for subject in path_dataset_bids.iterdir():
    if not (subject.is_dir() and subject.name.startswith('sub-')):
        continue
    
    oasis_id = subject.name.replace('sub-', '')
    
    # collect info at entry
    try:
        age_at_entry = demog.loc[demog['OASISID']==oasis_id, 'AgeatEntry'].values[0]
    except:
        age_at_entry = None
    
    try:
        sex = demog.loc[demog['OASISID']==oasis_id, 'GENDER'].values[0]
        sex = dict_sex2standard[sex]
    except:
        sex = None

    try:
        race = demog.loc[demog['OASISID']==oasis_id, 'race'].values[0]
        race = dict_race2standard[race]
    except:
        race = None
    
    # collect info for each session
    for session in subject.iterdir():
        if not (session.is_dir() and session.name.startswith('ses-')):
            continue
        
        days_from_entry = int(session.name.split('ses-d')[1])
        
        if age_at_entry != None:
            age = age_at_entry + days_from_entry/365
        else:
            age = None
        
        # approximate diagnosis at the current session
        filtered_df = diagnosis[diagnosis['OASISID']==oasis_id].copy()
        filtered_df['time_error'] = (filtered_df['days_to_visit'] - days_from_entry).abs()
        closest_row = filtered_df[filtered_df['time_error'] == filtered_df['time_error'].min()]
        
        if len(closest_row.index) == 0:
            dx = None
        else:
            if closest_row['NORMCOG'].values[0]==1:
                dx = 'normal'
            elif closest_row['DEMENTED'].values[0]==1:
                dx = 'dementia'
            else:
                dx = 'MCI'
        
        list_df_subject.append(subject.name)
        list_df_session.append(session.name)
        list_df_race.append(race)
        list_df_sex.append(sex)
        list_df_age.append(age)
        list_df_diagnosis.append(dx)

d = {'subject': list_df_subject,
     'session': list_df_session,
     'race': list_df_race,
     'sex': list_df_sex,
     'age': list_df_age,
     'diagnosis': list_df_diagnosis,
}
df = pd.DataFrame(data=d)
df.sort_values(by=['subject', 'age'], inplace=True)

df.to_csv('./data/subject_info/clean/OASIS3_info.csv', index=False)
