# Collect information of interest. 
# 
# Author: Chenyu Gao
# Date: Nov 15, 2023

import pandas as pd
from pathlib import Path

path_dataset_bids = Path('/nfs2/harmonization/BIDS/HCPA')
demog = pd.read_csv('./data/subject_info/raw/HCPA/HCPA_covars.csv', skiprows=[1])

# Dictionaries to standardize the data
dict_sex2standard = {
    'M': 'male',
    'F': 'female',
}
dict_race2standard = {
    'White': 'White',
    'Black or African American': 'Black or African American', 
    'Asian': 'Asian',
    'More than one race': 'More than one race',
    'Unknown or not reported': None,
    'American Indian/Alaska Native': 'American Indian or Alaska Native',
}
dict_dx2standard = {
    'Healthy Subject': 'normal',
}

# columns of interest
list_df_subject = []
list_df_session = []
list_df_race = []
list_df_sex = []
list_df_age = []
list_df_diagnosis = []

# Loop through subjects (this is a cross-sectional dataset)
for subject in path_dataset_bids.iterdir():
    if not (subject.is_dir() and subject.name.startswith('sub-')):
        continue
    subject_id = subject.name.replace('sub-', '')
    
    # race
    try:
        values = demog.loc[demog['src_subject_id']==subject_id, 'race'].values
        values = values[~pd.isna(values)]
        race = values[0]
        race = dict_race2standard[race]
    except:
        race = None

    # sex
    try:
        values = demog.loc[demog['src_subject_id']==subject_id, 'sex'].values
        values = values[~pd.isna(values)]
        sex = values[0]
        sex = dict_sex2standard[sex]
    except:
        sex = None

    # age
    try:
        values = demog.loc[demog['src_subject_id']==subject_id, 'interview_age'].values
        values = values[~pd.isna(values)]
        age = values[0] / 12
    except:
        age = None
    
    # diagnosis
    try:
        values = demog.loc[demog['src_subject_id']==subject_id, 'phenotype'].values
        values = values[~pd.isna(values)]
        dx = values[0]
        dx = dict_dx2standard[dx]
    except:
        dx = None
    
    list_df_subject.append(subject.name)
    list_df_session.append(None)
    list_df_race.append(race)
    list_df_sex.append(sex)
    list_df_age.append(age)
    list_df_diagnosis.append(dx)
    
# Save to csv
d = {'subject': list_df_subject,
     'session': list_df_session,
     'race': list_df_race,
     'sex': list_df_sex,
     'age': list_df_age,
     'diagnosis': list_df_diagnosis,
}
df = pd.DataFrame(data=d)
df.sort_values(by=['subject', 'age'], inplace=True)

df.to_csv('./data/subject_info/clean/HCPA_info.csv', index=False)


