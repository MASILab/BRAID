# Collect information of interest.
# 
# Author: Chenyu Gao
# Date: Nov 9, 2023

import json
import pandas as pd
from pathlib import Path

path_dataset_bids = Path('/nfs/masi/gaoc11/GDPR/BIDS/UKBB')
demog = pd.read_csv('./data/subject_info/raw/UKBB/ukbb_raw.csv')

# https://biobank.ctsu.ox.ac.uk/crystal/coding.cgi?id=9
dict_sex2standard = {
    0: "female",
    1: "male"
}
# https://biobank.ctsu.ox.ac.uk/crystal/coding.cgi?id=1001
dict_race2standard = {
    1: 'White',
    1001: 'British',
    2001: 'White and Black Caribbean',
    3001: 'Indian',
    4001: 'Caribbean',
    2: 'Mixed',
    1002: 'Irish',
    2002: 'White and Black African',
    3002: 'Pakistani',
    4002: 'African',
    3: 'Asian or Asian British',
    1003: 'Any other white background',
    2003: 'White and Asian',
    3003: 'Bangladeshi',
    4003: 'Any other Black background',
    4: 'Black or Black British',
    2004: 'Any other mixed background',
    3004: 'Any other Asian background',
    5: 'Chinese',
    6: 'Other ethnic group',
    -1: 'Do not know',
    -3:	'Prefer not to answer'
}

list_df_subject = []
list_df_race = []
list_df_sex = []
list_df_age = []
list_df_age_secondscan = []
list_df_CNS_controls_1 = []
list_df_CNS_controls_2 = []

# Loop through sessions we have, and collect information
for subject in path_dataset_bids.iterdir():
    if not (subject.is_dir() and subject.name.startswith('sub-')):
        continue
    
    # subject label for referencing
    eid = int(subject.name.replace('sub-',''))
    
    # race
    try:
        values = demog.loc[demog['eid']==eid, 'Race'].values
        values = values[~pd.isna(values)]
        race = values[0]
        race = dict_race2standard[race]
    except:
        race = None    
    
    # sex
    try:
        values = demog.loc[demog['eid']==eid, '31-0.0'].values
        values = values[~pd.isna(values)]
        sex = values[0]
        sex = dict_sex2standard[sex]
    except:
        sex = None
            
    # age
    try:
        values = demog.loc[demog['eid']==eid, 'AgeAtScan'].values
        values = values[~pd.isna(values)]
        age = values[0]
    except:
        age = None
    try:
        values = demog.loc[demog['eid']==eid, 'AgeAt2ndScan'].values
        values = values[~pd.isna(values)]
        age_secondscan = values[0]
    except:
        age_secondscan = None

    # CNS_controls_1
    try:
        values = demog.loc[demog['eid']==eid, 'CNS_controls_1'].values
        values = values[~pd.isna(values)]
        cns_control_1 = values[0]
    except:
        cns_control_1 = None
    # CNS_controls_2
    try:
        values = demog.loc[demog['eid']==eid, 'CNS_controls_2'].values
        values = values[~pd.isna(values)]
        cns_control_2 = values[0]
    except:
        cns_control_2 = None
        
    list_df_subject.append(subject.name)
    list_df_race.append(race)
    list_df_sex.append(sex)
    list_df_age.append(age)
    list_df_age_secondscan.append(age_secondscan)
    list_df_CNS_controls_1.append(cns_control_1)
    list_df_CNS_controls_2.append(cns_control_2)

# Save to csv
d = {'subject': list_df_subject,
     'race': list_df_race,
     'sex': list_df_sex,
     'age': list_df_age,
     'age_secondscan': list_df_age_secondscan,
     'CNS_control_1': list_df_CNS_controls_1,
     'CNS_control_2': list_df_CNS_controls_2
}
df = pd.DataFrame(data=d)
df.sort_values(by=['subject'], inplace=True)

df.to_csv('./data/subject_info/clean/UKBB_info.csv', index=False)
