# Collect information of interest.
# 
# Author: Chenyu Gao
# Date: Nov 9, 2023

import json
import pandas as pd
from pathlib import Path

path_dataset_bids = Path('/nfs2/harmonization/BIDS/WRAP')
demog = pd.read_csv('./data/subject_info/raw/WRAP/WRAP_Prepped_for_Harmonization_Fall2023_Release_Covariates_Only.csv')

dict_sex2standard = {
    1: 'male',
    2: 'female'
}

dict_race2standard = {
    1: 'American Indian or Alaska Native',
    2: 'Asian',
    3: 'Black or African American',	
    4: 'Native Hawaiian or Other Pacific Islander',	
    5: 'White',
    6: 'Other, Unknown, or More than one race'
}

dict_dx2standard = {
    1: 'No Cognitive Impairment',
    2: 'Mild Cognitive Impairment',	
    3: "Alzheimer's Dementia",
    'NA': 'Not available or Other Dementia (not AD)'
}

list_df_subject = []
list_df_session = []
list_df_race = []
list_df_sex = []
list_df_age = []
list_df_diagnosis = []

# Loop through subjects, and collect information
for subject in path_dataset_bids.iterdir():
    if not (subject.is_dir() and subject.name.startswith('sub-')):
        continue
    subject_id = subject.name.replace('sub-wrap','')
    
    # sex
    try:
        values = demog.loc[demog['Phenotype_ID']==subject_id, 'PHC_Sex'].values
        values = values[~pd.isna(values)]
        sex = values[0]
        sex = dict_sex2standard[sex]
    except:
        sex = None
    
    # race
    try:
        values = demog.loc[demog['Phenotype_ID']==subject_id, 'PHC_Race'].values
        values = values[~pd.isna(values)]
        race = values[0]
        race = dict_race2standard[race]
    except:
        race = None
    
    # collect info for each session
    for session in subject.iterdir():
        if not (session.is_dir() and session.name.startswith('ses-')):
            continue
        session_id = session.name.replace('ses-','')
        
        # age
        try:
            age = demog.loc[(demog['Phenotype_ID']==subject_id)&(demog['Session']==session_id), 'PHC_Age_Imaging'].values[0]
        except:
            age = None
            
        # diagnosis
        try:
            dx = demog.loc[(demog['Phenotype_ID']==subject_id)&(demog['Session']==session_id), 'PHC_Diagnosis'].values[0]
            dx = dict_dx2standard[dx]
        except:
            dx = None
            
        list_df_subject.append(subject.name)
        list_df_session.append(session.name)
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

df.to_csv('./data/subject_info/clean/WRAP_info.csv', index=False)
