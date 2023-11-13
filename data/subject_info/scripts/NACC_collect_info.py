# Collect information of interest.
# 
# Author: Chenyu Gao
# Date: Sept 4, 2023

import pandas as pd
from pathlib import Path

path_dataset_bids = Path('/nfs2/harmonization/BIDS/NACC')
df_info = pd.read_csv('./data/subject_info/raw/NACC/NACC_participants.tsv', sep='\t')

dict_sex2standard = {1: 'male', 2: 'female'}
dict_race2standard = {
    'NHW': 'White',
    'Other': 'Other',
    'NHB': 'Black or African American',
}
dict_dx2standard = {
    'MCI': 'MCI',
    'Control': 'normal',
    'Dementia': 'dementia',
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
        
        bids_id = f"{subject.name}_{session.name}"
        
        try:
            values = df_info.loc[df_info['BIDS']==bids_id, 'RACE'].values
            values = values[~pd.isna(values)]
            race = values[0]
            race = dict_race2standard[race]
        except:
            race = None
            
        try:
            values = df_info.loc[df_info['BIDS']==bids_id, 'SEX'].values
            values = values[~pd.isna(values)]
            sex = values[0]
            sex = dict_sex2standard[sex]
        except:
            sex = None
        
        try:
            values = df_info.loc[df_info['BIDS']==bids_id, 'AGE'].values
            values = values[~pd.isna(values)]
            age = values[0]
        except:
            age = None
            
        try:
            values = df_info.loc[df_info['BIDS']==bids_id, 'AD_Group'].values
            values = values[~pd.isna(values)]
            dx = values[0]
            dx = dict_dx2standard[dx]
        except:
            dx = None
        
        try:
            values = df_info.loc[df_info['BIDS']==bids_id, 'Diagnosis1'].values
            values = values[~pd.isna(values)]
            dx_detail = values[0]
        except:
            dx_detail = None
                    
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

df.to_csv('./data/subject_info/clean/NACC_info.csv', index=False)

# updated on Nov 13, 2023:
# verify the values according to the new csv we received
df = pd.read_csv('./data/subject_info/clean/NACC_info.csv')
df_new = pd.read_csv('./data/subject_info/raw/NACC/NACC_Prepped_for_Harmonization_Fall2023_Release_Covariates_Only.csv')

# The previous has column "diagnosis_detail" that we can use to identify the diagnosis
# dict_dx2previouscoding = {
#     1: 'normal',
#     2: 'MCI',	
#     3: "AD",
#     'NA': 'Not available or Other Dementia (not AD)'
# }

for _,row in df_new.iterrows():
    subject = "sub-{}".format(row['Phenotype_ID'])
    session = f"ses-{row['Session']}"
    age = row['PHC_Age_Imaging']
    # dx = row['PHC_Diagnosis']
    # try:
    #     dx = dict_dx2previouscoding[dx]
    # except:
    #     dx = None
        
    # Check if the session exists in BIDS
    if len(df.loc[(df['subject']==subject)&(df['session']==session), ].index) == 0:
        print("Extra sessions that do not exist in harmonization/BIDS")
        continue
    else:
        # verify age
        if df.loc[(df['subject']==subject)&(df['session']==session), 'age'].values.shape[0] == 0:
            print('Use the age in the new csv to fill the empty cell: {} {} {}'.format(subject, session, age))
            df.loc[(df['subject']==subject)&(df['session']==session), 'age'] = age
        else:
            if abs(df.loc[(df['subject']==subject)&(df['session']==session), 'age'].values[0] - age) > 0.2:
                print('Found large discrepancy in age. {} {} Previous: {}, Updated: {}'.format(subject, 
                                                                                               session, 
                                                                                               df.loc[(df['subject']==subject)&(df['session']==session), 'age'].values[0], 
                                                                                               age))
                df.loc[(df['subject']==subject)&(df['session']==session), 'age'] = age
            
        # # verify diagnosis
        # if df.loc[(df['subject']==subject)&(df['session']==session), 'diagnosis'].values.shape[0] == 0:
        #     print('Use the diagnosis in the new csv (could also be None) to fill the empty cell: {} {} {}'.format(subject, session, dx))
        #     df.loc[(df['subject']==subject)&(df['session']==session), 'diagnosis'] = dx
        # else:
        #     dx_previous = df.loc[(df['subject']==subject)&(df['session']==session), 'diagnosis'].values[0]
        #     if (dx_previous != dx) and (dx != None):
        #         print('Found different diagnosis for {} {}, previous: {}, updated: {}'.format(subject,session, dx_previous, dx))
        #         df.loc[(df['subject']==subject)&(df['session']==session), 'diagnosis'] = dx

# Save the updated csv
df.to_csv('./data/subject_info/clean/NACC_info.csv', index=False)