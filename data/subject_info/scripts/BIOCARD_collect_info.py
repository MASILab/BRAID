# Collect information of interest.
# 
# Author: Chenyu Gao
# Date: Aug 31, 2023

import json
import pandas as pd
from pathlib import Path

path_dataset_bids = Path('/nfs2/harmonization/BIDS/BIOCARD')

demog_1 = pd.read_excel('./data/subject_info/raw/BIOCARD/BIOCARD_External_Data_2022.08.08/BIOCARD_Demographics_Limited_Data_2022.05.10.xlsx')
demog_1 = demog_1[['JHUANONID', 'SEX', 'BIRTHYEAR', 'RACE']]
demog_2 = pd.read_excel('./data/subject_info/raw/BIOCARD/BIOCARD_Limited_Data_August_2021_data_freeze/BIOCARD_Demographics_Limited_Data_2021.12.13.xlsx')
demog_2 = demog_2[['JHUANONID', 'SEX', 'BIRTHYEAR', 'RACE']]
demog = pd.concat([demog_1, demog_2], ignore_index=True)
demog.drop_duplicates(inplace=True)

diagnosis_1 = pd.read_excel('./data/subject_info/raw/BIOCARD/BIOCARD_External_Data_2022.08.08/BIOCARD_DiagnosisData_Limited_2022.05.14.xlsx')
diagnosis_1 = diagnosis_1[['JHUANONID', 'MOFROMBL', 'STARTYEAR', 'BIRTHYR', 'DIAGNOSIS']]
diagnosis_2 = pd.read_excel('./data/subject_info/raw/BIOCARD/BIOCARD_Limited_Data_August_2021_data_freeze/BIOCARD_DiagnosisData_Limited_2021.11.01.xlsx')
diagnosis_2 = diagnosis_2[['JHUANONID', 'MOFROMBL', 'STARTYEAR', 'BIRTHYR', 'DIAGNOSIS']]
diagnosis = pd.concat([diagnosis_1, diagnosis_2], ignore_index=True)
diagnosis.drop_duplicates(inplace=True)

dict_sex2standard = {1:'male', 2:'female'}
dict_race2standard = {
    1: 'White',
    2: 'Black or African American',
    3: 'American Indian or Alaska Native',
    4: 'Native Hawaiian or Other Pacific Islander',
    5: 'Asian'
}
dict_dx2standard = {
    'NORMAL': 'normal',
    'IMPAIRED NOT MCI': 'impaired (not MCI)',
    'MCI': 'MCI',
    'DEMENTIA': 'dementia',
}

list_df_subject = []
list_df_session = []
list_df_sex = []
list_df_age = []
list_df_diagnosis = []
list_df_race = []

# Loop through sessions we have and collect information
for subject in path_dataset_bids.iterdir():
    if not (subject.is_dir() and subject.name.startswith('sub-')):
        continue
    
    JHUANONID = subject.name.split('sub-')[1]
    
    for session in subject.iterdir():
        if not (session.is_dir() and session.name.startswith('ses-')):
            continue
        
        # age
        current_year = int(session.name.split('ses-')[1][0:2]) + 2000
        if current_year >= 2023:
            print("Warning: data from the future!")
            break
        try:
            age = current_year - demog.loc[demog['JHUANONID']==JHUANONID, 'BIRTHYEAR'].values[0]
        except:
            age = None
            
        # sex
        try:
            sex = demog.loc[demog['JHUANONID']==JHUANONID, 'SEX'].values[0]
            sex = dict_sex2standard[sex]
        except:
            sex = None
            
        # race
        try:
            race = demog.loc[demog['JHUANONID']==JHUANONID, 'RACE'].values[0]
            race = dict_race2standard[race]
        except:
            race = None

        # diagnosis
        try:
            dx = diagnosis.loc[(diagnosis['JHUANONID']==JHUANONID)&(diagnosis['STARTYEAR']==current_year), 'DIAGNOSIS'].values[0]
            dx = dict_dx2standard[dx]
        except:
            dx = None
            
        list_df_subject.append(subject.name)
        list_df_session.append(session.name)
        list_df_sex.append(sex)
        list_df_age.append(age)
        list_df_diagnosis.append(dx)
        list_df_race.append(race)
        
d = {'subject': list_df_subject,
     'session': list_df_session,
     'sex': list_df_sex,
     'age': list_df_age,
     'diagnosis': list_df_diagnosis,
     'race': list_df_race}
df = pd.DataFrame(data=d)

df.to_csv('./data/subject_info/clean/BIOCARD_info.csv', index=False)
