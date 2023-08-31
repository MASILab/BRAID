from pathlib import Path
import pandas as pd
import json

path_dataset_bids = Path('/nfs2/harmonization/BIDS/ADNI_DTI')  # TODO: the folder will be renamed to ADNI

df_cognitive = pd.read_csv('./BRAID/data/subject_info/raw/ADNI/Assessments/Neuropsychological/ADSP_PHC_COGN_10_05_22_30Aug2023.csv')
df_mri_t1 = pd.read_csv('./BRAID/data/subject_info/raw/ADNI/ADNI_MRI_T1.csv')

# convert values of columns for easier referencing
with open('./BRAID/data/subject_info/scripts/ADNI_visit_LUT.json', 'r') as f:
    dict_adni_visit_lut = json.load(f)
df_mri_t1['subject_id'] = df_mri_t1['Subject ID'].str.split('_S_').str[-1]
df_mri_t1['visit_bids'] = df_mri_t1['Visit'].map(dict_adni_visit_lut)
df_mri_t1['study_date'] = pd.to_datetime(df_mri_t1['Study Date'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')
df_mri_t1 = df_mri_t1[['subject_id', 'Sex', 'visit_bids', 'Age', 'study_date']]

for subject in path_dataset_bids.iterdir():
    if not (subject.is_dir() and subject.name.startswith('sub-')):
        continue
    
    subject_id = subject.name.split('sub-')[1]
    
    for session in subject.iterdir():
        if not (session.is_dir() and session.name.startswith('ses-')):
            continue
        
        visit_id = session.name.split('ses-')[1]
        
        print(pd.unique(df_mri_t1.loc[(df_mri_t1['subject_id']==subject_id)&(df_mri_t1['visit_bids']==visit_id), 'study_date']))