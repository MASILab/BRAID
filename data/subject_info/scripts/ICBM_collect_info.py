# Collect information of interest.
# 
# Author: Chenyu Gao
# Date: Sept 4, 2023

import json
import pandas as pd
from pathlib import Path

path_dataset_bids = Path('/nfs2/harmonization/BIDS/ICBM')
demog = pd.read_excel('./data/subject_info/raw/ICBM/ICBM_Clinical_Data_18APR2012.xlsx',
                      header=0, usecols="A:F",skiprows=[1,2,192,193,194,195,196,197,198,199,200,201])

dict_sex2standard = {
    'Female': 'female',
    'Male': 'male'
}
dict_race2standard = {
    'White': 'White',
    'Unknown': None,
    'Black or African American': 'Black or African American',
    'Asian': 'Asian', 
    'Decline to State': None,
    'No Response': None,
    'American Native/Alaskan Indian and White': 'American Indian or Alaska Native',
    'Black': 'Black or African American',
    'Asian/ White': 'Asian',
    'Hispanic': 'Hispanic or Latino',
    'M': None
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

    subject_id = f"{subject.name.replace('sub-', '')[0:-4]}_{subject.name.replace('sub-', '')[-4:]}"
    
    try:
        values = demog.loc[demog['Subject ID']==subject_id, 'Age'].values
        values = values[~pd.isna(values)]
        age = values[0]
    except:
        age = None
    
    try:
        values = demog.loc[demog['Subject ID']==subject_id, 'Race'].values
        values = values[~pd.isna(values)]
        race = values[0]
        race = dict_race2standard[race]
    except:
        race = None
    
    try:
        values = demog.loc[demog['Subject ID']==subject_id, 'Gender'].values
        values = values[~pd.isna(values)]
        sex = values[0]
        sex = dict_sex2standard[sex]
    except:
        sex = None
    
    list_df_subject.append(subject.name)
    list_df_session.append(None)
    list_df_race.append(race)
    list_df_sex.append(sex)
    list_df_age.append(age)
    list_df_diagnosis.append('normal')
    
d = {'subject': list_df_subject,
     'session': list_df_session,
     'race': list_df_race,
     'sex': list_df_sex,
     'age': list_df_age,
     'diagnosis': list_df_diagnosis,
}
df = pd.DataFrame(data=d)
df.sort_values(by=['subject', 'age'], inplace=True)

df.to_csv('./data/subject_info/clean/ICBM_info.csv', index=False)
