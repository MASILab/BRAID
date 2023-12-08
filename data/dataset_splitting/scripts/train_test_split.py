"""
We've done QA and filtered out unwanted scans from the databank_dti.
And we've decided the age range we want the model to focus on.
Now, we split the filtered databank_dti into train and test sets.
We also split the train set into five folds for cross validation.
We store the data on the GDPR server.

logic:
1. We have a pool of scans that have passed QAs
2. pull subjects for test sets out of the pool
    - all cognitively impaired subjects will be used for testing
    - the entire ICBM dataset will be used for testing
    - For the rest of the datasets, pull out n subjects from each dataset,
      where n depends on the dataset (whether further analysis is possible). 
3. the rest of the subjects will be used for training
4. there must be scan-rescan data available in testing
5. there must be no overlap subjects between train and test sets
6. combining the train and test sets will lead to the original #rows

Author: Chenyu Gao
Date: Dec 7, 2023
"""

import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import KFold
from braid.utls import summarize_dataset

AGEMIN = 45
AGEMAX = 90
SEED = 1997

path_databank_dti = Path('/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_dti')  # all data including preprocessing
path_braid_data = Path('/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/braid_dataset')  # only preprocessed FA, MD for training and testing
path_cv_npy_root = Path('/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/cross_validation')  # array of dataset_subject for each fold

# Scan data remained after QAs
df_qa = pd.read_csv('/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/quality_assurance/databank_dti_after_pngqa_after_adspqa.csv')
df_qa = df_qa.loc[df_qa['age'].notnull(), ]
df_qa['dataset_subject'] = df_qa['dataset'] + '_' + df_qa['subject']
df_pool = df_qa.copy()

# Assign ICBM and all cognitively impaired subjects to test set
subjects = df_pool.loc[(df_pool['dataset']=='ICBM')|(df_pool['control_label']==0), 'dataset_subject'].unique()
print(f'Assigning {subjects.shape[0]} subjects (ICBM and all cognitively impaired) to test set...')
df_test = df_pool.loc[df_pool['dataset_subject'].isin(subjects), ].copy()
df_pool.drop(df_pool.loc[df_pool['dataset_subject'].isin(subjects), ].index, inplace=True)
print(f'Removed these {subjects.shape[0]} subjects from the pool. #Remaining scans: {df_pool.shape[0]}\n')

# Assign n subjects (whose ages fall within the chosen range) from each dataset to test set
num_subjects_test = {
    'BIOCARD': 50,
    'UKBB': 50,
    'NACC': 50,
    'HCPA': 50,
    'OASIS4': 5,
    'BLSA': 100,  # for scan-rescan experiment
    'OASIS3': 100,  # for cognitive performance score
    'ADNI': 300,  # for cognitive performance score
    'VMAP': 50,
    'WRAP': 50,
    'ROSMAPMARS': 50,
}
for dataset in num_subjects_test.keys():
    print(f'Assigning {num_subjects_test[dataset]} subjects from {dataset} to test set...')
    subjects = df_pool.loc[(df_pool['dataset']==dataset) & (df_pool['age']>=AGEMIN) & (df_pool['age']<=AGEMAX), 'dataset_subject'].unique()
    np.random.seed(SEED)
    subjects = np.random.choice(subjects, size=num_subjects_test[dataset], replace=False)
    df_test = pd.concat([df_test,
                         df_pool.loc[df_pool['dataset_subject'].isin(subjects), ]], 
                        ignore_index=True)
    df_pool.drop(df_pool.loc[df_pool['dataset_subject'].isin(subjects), ].index, inplace=True)
    print(f'Removed these {subjects.shape[0]} subjects from the pool. #Remaining scans: {df_pool.shape[0]}\n')

# Assign remaining subjects whose ages fall within the chosen range to train set
subjects = df_pool.loc[(df_pool['age']>=AGEMIN) & (df_pool['age']<=AGEMAX), 'dataset_subject'].unique()
print(f'Assigning {subjects.shape[0]} subjects to train set...')
df_train = df_pool.loc[df_pool['dataset_subject'].isin(subjects), ].copy()
df_pool.drop(df_pool.loc[df_pool['dataset_subject'].isin(subjects), ].index, inplace=True)
print(f'Removed these {subjects.shape[0]} subjects from the pool. #Remaining scans: {df_pool.shape[0]}\n')

# Assign the rest of the scans (age out of range) to test set, just for fun
for _,row in df_pool.iterrows():
    if row['age'] >= AGEMIN and row['age'] <= AGEMAX:
        raise ValueError('There are still scans whose ages fall within the chosen range.')
df_test = pd.concat([df_test, df_pool], ignore_index=True)

# Verify there is no information leakage between train and test sets
for _,row in df_test.iterrows():
    if row['dataset_subject'] in df_train['dataset_subject'].values:
        raise ValueError('There is information leakage between train and test sets.')

print(f'Dataset splitting completed. Saving csv files to GDPR...\n')
df_train.to_csv('/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/braid_train.csv', index=False)
df_test.to_csv('/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/braid_test.csv', index=False)

print('\n============ Training Set ============')
print('------------ Overall ------------')
summarize_dataset(df_train.copy())
for dataset in df_train['dataset'].unique():
    print(f"------------ {dataset} ------------")
    summarize_dataset(df_train.loc[df_train['dataset'] == dataset, ].copy())

print('\n============ Testing Set ============')
print('------------ Overall ------------')
summarize_dataset(df_test.copy())
for dataset in df_test['dataset'].unique():
    print(f"------------ {dataset} ------------")
    summarize_dataset(df_test.loc[df_test['dataset'] == dataset, ].copy())


# # Data transfer
# print(f'\nTransferring data to {path_braid_data}\n')

# df_all = pd.concat([df_train, df_test], ignore_index=True)
# if df_all.shape[0] != df_qa.shape[0]:
#     raise ValueError('The combined dataset does not have the same #rows as the original dataset.')

# for _,row in tqdm(df_all.iterrows(), total=df_all.shape[0]):
#     fa = path_databank_dti / row['dataset'] / row['subject'] / row['session'] / f"scan-{row['scan']}" / 'final' / 'fa_skullstrip_MNI152.nii.gz'
#     md = path_databank_dti / row['dataset'] / row['subject'] / row['session'] / f"scan-{row['scan']}" / 'final' / 'md_skullstrip_MNI152.nii.gz'
    
#     if not (fa.is_file() and md.is_file()):
#         raise ValueError(f'The following files do not exist: {fa}, {md}')
    
#     fa_dst = path_braid_data / row['dataset'] / row['subject'] / row['session'] / f"scan-{row['scan']}" / 'fa_skullstrip_MNI152.nii.gz'
#     md_dst = path_braid_data / row['dataset'] / row['subject'] / row['session'] / f"scan-{row['scan']}" / 'md_skullstrip_MNI152.nii.gz'
    
#     subprocess.run(['mkdir', '-p', str(fa_dst.parent)])
#     subprocess.run(['rsync', '-a', str(fa), str(fa_dst)])
#     subprocess.run(['rsync', '-a', str(md), str(md_dst)])

# 5 Folds for cross validation
subjects = df_train['dataset_subject'].unique()
subject_indices = np.arange(subjects.shape[0])

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

for i, (train_indices, val_indices) in enumerate(kf.split(subject_indices)):
    subjects_train, subjects_val = subjects[train_indices], subjects[val_indices]
    
    # Save to .npy
    npy_train = path_cv_npy_root / f'subjects_fold_{i+1}_train.npy'
    npy_val = path_cv_npy_root / f'subjects_fold_{i+1}_val.npy'
    
    np.save(npy_train, subjects_train)
    np.save(npy_val, subjects_val)

print(f'5-fold cross-validation completed. Saving .npy files to {path_cv_npy_root}\n')
