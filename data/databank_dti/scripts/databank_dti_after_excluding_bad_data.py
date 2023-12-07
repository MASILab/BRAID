"""
We performed a QA on the PNG images and deleted the PNG images that show bad FA and MD images.
The correpsonding FA and MD images of the remaining PNG images are considered as passing the QA.

Also, for ADNI, BIOCARD, NACC, OASIS3/4, ROSMAPMARS, and WRAP, we performed a QA on the
PreQual, SLANT, and white matter registration results during the ADSP project.
We generated a list of PreQual paths that passed the QA.

Now we try to combine the QA results to generate the final list of FA, MD images to use.

Author: Chenyu Gao
Date: Dec 6, 2023
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from braid.utls import summarize_dataset

# All subject info for reference
df_master = pd.read_csv('/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti.csv')
df_master = df_master.loc[df_master['age'].notnull(), ]

list_all_datasets = df_master['dataset'].unique()

# Data remained after the QA on the PNG images
path_png_root = Path('./data/databank_dti/quality_assurance/QA_databank_dti_done')

list_df_dataset = []
list_df_subject = []
list_df_session = []
list_df_scan = []
list_df_sex = []
list_df_race = []
list_df_age = []
list_df_control_label = []

for dataset in path_png_root.iterdir():
    if not dataset.name in list_all_datasets:
        print(f'Warning: {dataset.name} not following original naming convention')
        continue
    
    for fn in dataset.iterdir():
        if fn.name.endswith('.png'):
            # parse the filename
            subject = fn.stem.split('_')[0]
            session = fn.stem.split('_')[1]
            scan = int(fn.stem.split('_')[2].replace('scan-',''))
            
            # locate the corresponding row in the master spreadsheet
            if dataset.name in ['UKBB', 'HCPA', 'ICBM']:
                row = df_master.loc[(df_master['dataset'] == dataset.name) & (df_master['subject'] == subject) & (df_master['scan'] == scan)]
            else:
                row = df_master.loc[(df_master['dataset'] == dataset.name) & (df_master['subject'] == subject) & (df_master['session'] == session) & (df_master['scan'] == scan)]
                
            if row.shape[0] == 0:
                print(f'Warning: no row was found for {fn.name}')
                continue
            
            # retrieve subject information
            sex = row['sex'].item()
            race = row['race_simple'].item()
            age = row['age'].item()
            control_label = row['control_label'].item()
                        
            list_df_dataset.append(dataset.name)
            list_df_subject.append(subject)
            list_df_session.append(session)
            list_df_scan.append(scan)
            list_df_sex.append(sex)
            list_df_race.append(race)
            list_df_age.append(age)
            list_df_control_label.append(control_label)

df_png = pd.DataFrame({'dataset': list_df_dataset,
                       'subject': list_df_subject,
                       'session': list_df_session,
                       'scan': list_df_scan,
                       'sex': list_df_sex,
                       'race_simple': list_df_race,
                       'age': list_df_age,
                       'control_label': list_df_control_label})

df_png.to_csv('./data/databank_dti/quality_assurance/databank_dti_after_pngqa.csv', index=False)

# Print out databank summary
df_png = pd.read_csv('./data/databank_dti/quality_assurance/databank_dti_after_pngqa.csv')
print('============Summary of the databank after the PNG QA============')
print('------------Overall------------')
summarize_dataset(df_png.copy())

for dataset in df_png['dataset'].unique():
    print(f"------------{dataset}------------")
    summarize_dataset(df_png.loc[df_png['dataset'] == dataset, ].copy())


# Data remained after the QA on the PNG images AND the ADSP QA
with open("./data/databank_dti/quality_assurance/QA_from_ADSP/step4_whitelist.txt", "r") as file:
    list_prequal_pass_qa = file.read().splitlines()

# create dataframe of whitelist
list_df_dataset = []
list_df_subject = []
list_df_session = []

for prequal in list_prequal_pass_qa:
    dataset = prequal.split('/')[4]
    subject = prequal.split('/')[6]
    session = prequal.split('/')[7]
    
    list_df_dataset.append(dataset)
    list_df_subject.append(subject)
    list_df_session.append(session)
    
df_adsp_whitelist = pd.DataFrame({
    'dataset': list_df_dataset,
    'subject': list_df_subject,
    'session': list_df_session
})

# standardize dataset names
df_adsp_whitelist['dataset'] = df_adsp_whitelist['dataset'].str.replace('ADNI_DTI', 'ADNI')

print('Looping through the remaining samples after the PNG QA...\n')
for i, row in df_png.iterrows():
    if not row['dataset'] in df_adsp_whitelist['dataset'].unique():
        continue
    else:
        if df_adsp_whitelist.loc[(df_adsp_whitelist['dataset'] == row['dataset']) 
                                 & (df_adsp_whitelist['subject'] == row['subject']) 
                                 & (df_adsp_whitelist['session'] == row['session'])].shape[0] == 0:
            print(f"\tdropped {row['dataset']} {row['subject']} {row['session']} {row['scan']}")
            df_png.drop(index=i, inplace=True)
df_png.to_csv('./data/databank_dti/quality_assurance/databank_dti_after_pngqa_after_adspqa.csv', index=False)

# Print out databank summary
df_png = pd.read_csv('./data/databank_dti/quality_assurance/databank_dti_after_pngqa_after_adspqa.csv')
print('============Summary of the databank after the PNG QA AND the ADSP QA============')
print('------------Overall------------')
summarize_dataset(df_png.copy())

for dataset in df_png['dataset'].unique():
    print(f"------------{dataset}------------")
    summarize_dataset(df_png.loc[df_png['dataset'] == dataset, ].copy())
