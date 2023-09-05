# Collect information of interest.
# 
# Author: Chenyu Gao
# Date: Sept 5, 2023

import json
import pandas as pd
from pathlib import Path

path_dataset_bids = Path('/nfs2/harmonization/BIDS/VMAP')

demog = pd.read_csv('./data/subject_info/raw/VMAP/Multisite_dMRI_Harmonized_all_tracts_10-10-22.csv')
dict_sex2standard = {
    1: 'male',
    2: 'female'
}
dict_dx2standard = {
    1: 'normal',
    2: 'MCI',
    3: 'dementia'
}
dict_race2standard = {
    'White': 'White',
    'Other': 'Other',
    'Black': 'Black or African American',
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
    
    subject_id = f"{subject.name.split('sub-')[1]}_VMAP"
    
    for session in subject.iterdir():
        if not (session.is_dir() and session.name.startswith('ses-')):
            continue
        
        session_id = session.name.split('x')[1]

        try:
            values = demog.loc[demog['Subject_Study']==subject_id, 'sex'].values
            values = values[~pd.isna(values)]
            sex = values[0]
            sex = dict_sex2standard[sex]
        except:
            sex = None
        
        try:
            values = demog.loc[(demog['Subject_Study']==subject_id), 'raceethnicity'].values
            values = values[~pd.isna(values)]
            race = values[0]
            race = dict_race2standard[race]
        except:
            race = None

        try:
            values = demog.loc[(demog['Subject_Study']==subject_id)&(demog['Session']==session_id), 'Age'].values
            values = values[~pd.isna(values)]
            age = values[0]
        except:
            age = None
        
        try:
            values = demog.loc[(demog['Subject_Study']==subject_id)&(demog['Session']==session_id), 'dx'].values
            values = values[~pd.isna(values)]
            dx = values[0]
            dx = dict_dx2standard[dx]
        except:
            dx = None
        
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

df.to_csv('./data/subject_info/clean/VMAP_info.csv', index=False)
