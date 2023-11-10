# Collect information of interest.
# 
# Author: Chenyu Gao
# Date: Aug 31, 2023

import json
import pandas as pd
from pathlib import Path

path_dataset_bids = Path('/nfs2/harmonization/BIDS/ADNI_DTI')  # TODO: the folder will be renamed to ADNI

df_mri_t1 = pd.read_csv('./data/subject_info/raw/ADNI/ADNI_MRI_T1.csv')  # already contain most of the information we need
df_cognitive = pd.read_csv('./data/subject_info/raw/ADNI/Assessments/Neuropsychological/ADSP_PHC_COGN_10_05_22_30Aug2023.csv')  # specific for cognitive info
df_w_race = pd.read_csv('./data/subject_info/raw/ADNI/Subject_Characteristics/Subject_Demographics/PTDEMOG_30Aug2023.csv')

# convert values of columns for easier referencing
with open('./data/subject_info/scripts/ADNI_visit_LUT.json', 'r') as f:
    dict_adni_visit_lut = json.load(f)
df_mri_t1['subject_id'] = df_mri_t1['Subject ID'].str.split('_S_').str[-1]
df_mri_t1['visit_bids'] = df_mri_t1['Visit'].map(dict_adni_visit_lut)
df_mri_t1['study_date'] = pd.to_datetime(df_mri_t1['Study Date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
df_mri_t1 = df_mri_t1[['subject_id', 'visit_bids', 'study_date', 'Sex', 'Age', 'Research Group']]
df_cognitive['EXAMDATE'] = pd.to_datetime(df_cognitive['EXAMDATE'], format='%Y-%m-%d')

# LUT
dict_diagnosis = {
    1: 'normal',
    2: 'MCI',
    3: "AD"
}
dict_sex = {
    'M': 'male',
    'F': 'female',
    1: 'male',
    2: 'female'
}
dict_race2standard = {   
    1: 'American Indian or Alaska Native',
    2: 'Asian',
    3: 'Native Hawaiian or Other Pacific Islander',
    4: 'Black or African American',
    5: 'White',
    6: 'More than one race',
    7: None
}

# record
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
    
    subject_id = subject.name.split('sub-')[1]
    
    for session in subject.iterdir():
        if not (session.is_dir() and session.name.startswith('ses-')):
            continue
        
        visit_id = session.name.split('ses-')[1]
        
        # collect age, sex, diagnosis, study_date (for additional mapping if necessary)
        try:
            values = df_mri_t1.loc[(df_mri_t1['subject_id']==subject_id)&(df_mri_t1['visit_bids']==visit_id), 'Age'].values
            values = values[~pd.isna(values)]
            age = values[0]
        except:
            age = None
            
        try:
            values = df_mri_t1.loc[(df_mri_t1['subject_id']==subject_id)&(df_mri_t1['visit_bids']==visit_id), 'Sex'].values
            values = values[~pd.isna(values)]
            sex = values[0]
            sex = dict_sex[sex]
        except:
            sex = None
        
        try:
            values = df_mri_t1.loc[(df_mri_t1['subject_id']==subject_id)&(df_mri_t1['visit_bids']==visit_id), 'Research Group'].values
            values = values[~pd.isna(values)]
            diagnosis_group = values[0]
        except:
            diagnosis_group = None
        
        try:
            values = df_mri_t1.loc[(df_mri_t1['subject_id']==subject_id)&(df_mri_t1['visit_bids']==visit_id), 'study_date'].values
            values = values[~pd.isna(values)]
            study_date = values[0]
        except:
            study_date = None
        
        try:
            race = df_w_race.loc[(df_w_race['RID']==int(subject_id))&(df_w_race['PTRACCAT']>0), 'PTRACCAT'].values[0]
            race = dict_race2standard[race]
        except:
            race = None
            
        # if study_date is available, use it to get missing info
        if study_date != None:
            study_date = pd.to_datetime(study_date, format='%Y-%m-%d')
            time_difference = abs(study_date - df_cognitive['EXAMDATE'])
            
            if age == None:
                try:
                    values = df_cognitive.loc[(df_cognitive['RID']==int(subject_id)) & (time_difference < pd.Timedelta(days=90)), 'PHC_AGE'].values
                    values = values[~pd.isna(values)]
                    age = values[0]
                    print("Use PHC_AGE to approximate age (with an error less than 90 days)")
                except:
                    age = None
            
            if sex == None:
                try:
                    values = df_cognitive.loc[(df_cognitive['RID']==int(subject_id)) & (time_difference < pd.Timedelta(days=90)), 'PHC_Sex'].values
                    values = values[~pd.isna(values)]
                    sex = values[0]
                    sex = dict_sex[sex]
                    print("Use PHC_Sex as sex (assuming the sex never changes)")
                except:
                    sex = None
            
            # Prefer cognitive info from ADSP_PHC_COGN_10_05_22_30Aug2023.csv
            try:
                values = df_cognitive.loc[(df_cognitive['RID']==int(subject_id)) & (time_difference < pd.Timedelta(days=90)), 'PHC_Diagnosis'].values
                values = values[~pd.isna(values)]
                diagnosis = values[0]
                diagnosis = dict_diagnosis[diagnosis]
                # print("Use PHC_Diagnosis as diagnosis")
            except:
                diagnosis = diagnosis_group
        
        # Record
        list_df_subject.append(subject.name)
        list_df_session.append(session.name)
        list_df_sex.append(sex)
        list_df_age.append(age)
        list_df_diagnosis.append(diagnosis)
        list_df_race.append(race)
        
d = {'subject': list_df_subject,
     'session': list_df_session,
     'sex': list_df_sex,
     'age': list_df_age,
     'diagnosis': list_df_diagnosis,
     'race': list_df_race}
df = pd.DataFrame(data=d)

df.to_csv('./data/subject_info/clean/ADNI_info.csv', index=False)

# Verify the values according to the new csv we received
df = pd.read_csv('./data/subject_info/clean/ADNI_info.csv')
df_new = pd.read_csv('./data/subject_info/raw/ADNI/ADNI_Prepped_for_Harmonization_Fall2023_Release_Covariates_Only.csv')

dict_dx2previouscoding = {
    1: 'normal',
    2: 'MCI',	
    3: "AD",
    'NA': 'Not available or Other Dementia (not AD)'
}

for _,row in df_new.iterrows():
    subject = "sub-{0:04d}".format(row['Phenotype_ID'])
    session = f"ses-{row['Session']}"
    age = row['PHC_Age_Imaging']
    dx = row['PHC_Diagnosis']
    try:
        dx = dict_dx2previouscoding[dx]
    except:
        dx = None
        
    # Check if the session exists in BIDS
    if len(df.loc[(df['subject']==subject)&(df['session']==session), ].index) == 0:
        print("Extra sessions that do not exist in harmonization/BIDS")
        continue
    else:
        # verify age
        if df.loc[(df['subject']==subject)&(df['session']==session), 'age'].values.shape[0] == 0:
            print('Use the age in the new csv to fill the empty cell: ')
            print(subject, session, age)
            df.loc[(df['subject']==subject)&(df['session']==session), 'age'] = age
        else:
            if abs(df.loc[(df['subject']==subject)&(df['session']==session), 'age'].values[0] - age) > 0.2:
                print('Found large discrepancy in age:')
                print(subject, session, age)
                df.loc[(df['subject']==subject)&(df['session']==session), 'age'] = age
            
        # verify diagnosis
        if df.loc[(df['subject']==subject)&(df['session']==session), 'diagnosis'].values.shape[0] == 0:
            print('Use the diagnosis in the new csv (could also be None) to fill the empty cell: ')
            print(subject, session, dx)
            df.loc[(df['subject']==subject)&(df['session']==session), 'diagnosis'] = dx
        else:
            dx_previous = df.loc[(df['subject']==subject)&(df['session']==session), 'diagnosis'].values[0]
            if (dx_previous != dx) and (dx != None):
                print('Found different diagnosis for {} {}, previous: {}, updated: {}'.format(subject,session, dx_previous, dx))
                df.loc[(df['subject']==subject)&(df['session']==session), 'diagnosis'] = dx

# Save the updated csv
df.to_csv('./data/subject_info/clean/ADNI_info.csv', index=False)