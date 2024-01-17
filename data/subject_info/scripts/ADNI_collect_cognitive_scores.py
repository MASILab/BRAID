# This script is written after the ADNI_collect_info.py
# It collects the memory score, executive function score, language score, visuospatial score.
# If a score is missing from that visit, it will use the value from the closest visit.
# (which introduces some error)
# 
# Author: Chenyu Gao
# Date: Jan 16, 2024

import pandas as pd
from tqdm import tqdm

df = pd.read_csv('./data/subject_info/clean/ADNI_info.csv')
coginfo = pd.read_csv('./data/subject_info/raw/ADNI/Assessments/Neuropsychological/ADSP_PHC_COGN_10_05_22_30Aug2023.csv')

# collect the following extra rows
df['memory_score'] = None
df['executive_function_score'] = None
df['language_score'] = None
df['visuospatial_score'] = None
ct_mem = 0
ct_exf = 0
ct_lan = 0
ct_vsp = 0

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    subject = row['subject']
    subject_id = subject.split('sub-')[1]
    age = row['age']

    coginfo_filtered = coginfo.loc[coginfo['RID']==int(subject_id), ].copy()
    coginfo_filtered['diff'] = abs(coginfo_filtered['PHC_AGE'] - age)
    coginfo_filtered = coginfo_filtered.sort_values('diff')

    # memory score
    try:
        values = coginfo_filtered['PHC_MEM'].values
        values = values[~pd.isna(values)]
        df.loc[index, 'memory_score'] = values[0]
    except:
        ct_mem += 1
    
    # executive function score
    try:
        values = coginfo_filtered['PHC_EXF'].values
        values = values[~pd.isna(values)]
        df.loc[index, 'executive_function_score'] = values[0]
    except:
        ct_exf += 1
    
    # language score
    try:
        values = coginfo_filtered['PHC_LAN'].values
        values = values[~pd.isna(values)]
        df.loc[index, 'language_score'] = values[0]
    except:
        ct_lan += 1
    
    # visuospatial score
    try:
        values = coginfo_filtered['PHC_VSP'].values
        values = values[~pd.isna(values)]
        df.loc[index, 'visuospatial_score'] = values[0]
    except:
        ct_vsp += 1

# outputs
print(f"Collection complete. Missing {ct_mem} memory scores, {ct_exf} executive function scores, {ct_lan} language scores, {ct_vsp} visuospatial scores.")
df.to_csv('./data/subject_info/clean/ADNI_info_w_approximate_cogn_scores.csv', index=False)