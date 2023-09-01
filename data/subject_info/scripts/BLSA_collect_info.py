# Collect information of interest.
# 
# Author: Chenyu Gao
# Date: Sept 1, 2023

import json
import pandas as pd
from pathlib import Path

path_dataset_bids = Path('/nfs2/harmonization/BIDS/BLSA')
demog = pd.read_csv('./data/subject_info/raw/BLSA/BLSA_cohorts.csv')

dict_sex2standard = {1:'male', 0:'female'}
dict_race2standard = {
    'White': 'White',
    'Black': 'Black or African American',
    'American Indian or Alaska Native': 'American Indian or Alaska Native',
    'Other NonWhite': None,
    'Other Asian or Other Pacific Islander': None,
    'Chinese': 'Asian',
    'Not Classifiable': None,
    'Japanese': 'Asian',
    'Filipino': 'Asian',
}
dict_dx2standard = {
    1: 'dementia',
    0.5: 'MCI', 
    -0.5: 'impaired (not MCI)', 
    0: 'normal'
}
# dict_dxtype2standard = {
#     0: 'normal',
#     1: 'Definite AD',
#     2: 'Probable AD',
#     3: 'Possible AD',
#     4: 'Vascular dementia',
#     5: 'Consistent w/ AD',
#     6: 'Depression',
#     7: 'Alcohol abuse',
#     8: 'PD',
#     9: None,
#     10: 'Other primary DX',
#     11: 'Other secondary DX',
#     12: 'Normal pressure hydrocephalus',
#     13: 'Hippocampal sclerosis',
#     14: 'Dementia w/ lewy body',
#     15: 'Frontotemporal dementia',
#     17: 'Vascular without dementia',
#     18: 'Other Dementia',
#     19: None
# }

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
    
    for session in subject.iterdir():
        if not (session.is_dir() and session.name.startswith('ses-')):
            continue
        
        label = "BLSA_{}_{}-{}_{}".format(subject.name.split('BLSA')[1],
                                          session.name.split('ses-')[1][0:2],
                                          session.name.split('ses-')[1][2],
                                          session.name.split('scanner')[1].split('_')[0]
        )
                    
        # age, sex, race, diagnosis
        try:
            values = demog.loc[demog['labels']==label, 'Age'].values
            values = values[~pd.isna(values)]
            age = values[0]
        except:
            age = None
        
        try:
            values = demog.loc[demog['labels']==label, 'sex'].values
            values = values[~pd.isna(values)]
            sex = values[0]
            sex = dict_sex2standard[sex]
        except:
            sex = None
            
        try:
            values = demog.loc[demog['labels']==label, 'race'].values
            values = values[~pd.isna(values)]
            race = values[0]
            race = dict_race2standard[race]
        except:
            race = None
        
        try:
            values = demog.loc[demog['labels']==label, 'dxatvi'].values
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
     'sex': list_df_sex,
     'age': list_df_age,
     'diagnosis': list_df_diagnosis,
     'race': list_df_race}
df = pd.DataFrame(data=d)
df.sort_values(by=['subject', 'age'], inplace=True)

df.to_csv('./data/subject_info/clean/BLSA_info.csv', index=False)
