""" Extract data useful for training and testing from databank_t1w.
Run this on hickory.
"""


import os
import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def set_up_traintestset_symlink(
    databank_t1w_dir = '/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_t1w',
    traintestset_dir = '/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/T1wAgePredict',
    suffix = '_T1w_MNI152_Warped_crop_downsample.nii.gz'
):
    
    for root, dirs, files in os.walk(databank_t1w_dir):
        for fn in files:
            if fn.endswith(suffix):
                fn_target = os.path.join(root, fn)
                fn_link = fn_target.replace(databank_t1w_dir, traintestset_dir)
                
                subprocess.run(['mkdir', '-p', os.path.dirname(fn_link)])
                subprocess.run(['ln', '-s', fn_target, fn_link])


def train_test_split_following_braid(
    braid_train_csv,
    braid_test_csv,
    databank_t1w_csv,
    t1wagepredict_train_csv,
    t1wagepredict_test_csv,
    traintestset_dir = '/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/T1wAgePredict',
    check_existence = True
):
    # subject-level train-test splitting of BRAID
    braid_train = pd.read_csv(braid_train_csv)
    braid_test = pd.read_csv(braid_test_csv)
    dataset_subjects_train = braid_train['dataset_subject'].tolist()
    dataset_subjects_test = braid_test['dataset_subject'].tolist()
    
    # use the same subject-level splitting for T1wAgePredict
    df = pd.read_csv(databank_t1w_csv)
    df = df.loc[df['age'].notnull(), ]
    df['dataset_subject'] = df['dataset'] + '_' + df['subject']
    df['session'] = df['session'].fillna('ses-1')
    df = df[['dataset','subject','session','scan','sex','race_simple','age','control_label','dataset_subject']]
    
    if check_existence:
        print(f"Total number of samples before file existence check: {len(df.index)}")
        data_root = Path(traintestset_dir)
        
        for i, row in tqdm(df.iterrows(), total=len(df.index), desc='Checking file existence'):
            t1w = data_root / row['dataset'] / row['subject'] / row['session'] / f"scan-{row['scan']}" / f"{row['dataset']}_{row['subject']}_{row['session']}_scan-{row['scan']}_T1w_MNI152_Warped_crop_downsample.nii.gz"
            
            if not t1w.is_file():
                print(f"File not found. drop from train/test sets: {t1w}")
                df.drop(i, inplace=True)
        print(f"Total number of samples after removing non-existing ones: {len(df.index)}")
    
    df_train = df.loc[df['dataset_subject'].isin(dataset_subjects_train), ]
    df_test = df.loc[df['dataset_subject'].isin(dataset_subjects_test), ]    
    
    # save to csv
    df_train.to_csv(t1wagepredict_train_csv, index=False)
    df_test.to_csv(t1wagepredict_test_csv, index=False)


if __name__ == "__main__":

    # set_up_traintestset_symlink()
    
    train_test_split_following_braid(
        braid_train_csv='/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/braid_train.csv',
        braid_test_csv='/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/braid_test.csv',
        databank_t1w_csv='/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_t1w.csv',
        t1wagepredict_train_csv='/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/t1wagepredict_train.csv',
        t1wagepredict_test_csv='/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/t1wagepredict_test.csv',
    )