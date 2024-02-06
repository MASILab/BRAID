""" 1) Filter databank with QA results. 
2) Split data into training and testing sets following BRAID.
3) Create symlinks for the data.
Run this on hickory.
"""

import os
import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool


def filter_qa_results(df, qa_results_csv):
    df_filtered = df.copy()

    qa = pd.read_csv(qa_results_csv)
    list_rejected_t1w = qa.loc[qa['qa']=='rejected', 't1w'].tolist()

    for t1w in list_rejected_t1w:
        dataset = t1w.split('/')[-1].split('_')[0]
        subject = t1w.split('/')[-1].split('_')[1]
        session = t1w.split('/')[-1].split('_')[2]
        scan = int(t1w.split('/')[-1].split('_')[3].replace('scan-', ''))

        df_filtered = df_filtered.loc[
            (df_filtered['dataset'] != dataset) |
            (df_filtered['subject'] != subject) |
            (df_filtered['session'] != session) |
            (df_filtered['scan'] != scan)
        ]
    print(f"Before QA: {len(df.index)}\nAfter QA: {len(df_filtered.index)}")
    
    return df_filtered


def train_test_split_following_braid(
    braid_train_csv,
    braid_test_csv,
    df_filtered,
    t1wagepredict_train_csv,
    t1wagepredict_test_csv,
    databank_t1w_dir,
    suffix,
    check_existence = True,
):
    # Sessions used in BRAID
    braid_train = pd.read_csv(braid_train_csv)
    braid_test = pd.read_csv(braid_test_csv)
    subject_train = braid_train['dataset_subject'].tolist()
    subject_test = braid_test['dataset_subject'].tolist()
    
    # Use the same subjects and splitting of train/test for T1wAgePredict
    df = df_filtered.copy()
    df = df.loc[df['age'].notnull(), ]
    df['session'] = df['session'].fillna('ses-1')
    df['dataset_subject'] = df['dataset'] + '_' + df['subject']
    
    if check_existence:
        print(f"Total number of samples before file existence check: {len(df.index)}")
        databank_t1w_dir = Path(databank_t1w_dir)
        
        for i, row in tqdm(df.iterrows(), total=len(df.index), desc='Checking file existence'):
            t1w = databank_t1w_dir / row['dataset'] / row['subject'] / row['session'] / f"scan-{row['scan']}" / f"{row['dataset']}_{row['subject']}_{row['session']}_scan-{row['scan']}{suffix}"
            if not t1w.is_file():
                df.drop(i, inplace=True)

        print(f"Total number of samples after removing non-existing ones: {len(df.index)}")
    
    columns_to_keep = ['dataset','subject','session','scan','sex','race_simple','age','control_label','dataset_subject']
    df_train = df.loc[df['dataset_subject'].isin(subject_train), columns_to_keep]
    df_test = df.loc[df['dataset_subject'].isin(subject_test), columns_to_keep]
    
    # save to csv
    df_train.to_csv(t1wagepredict_train_csv, index=False)
    df_test.to_csv(t1wagepredict_test_csv, index=False)


def create_symlink_job_tuple(t1wagepredict_train_csv, t1wagepredict_test_csv, databank_t1w_dir, suffix, traintestset_dir):
    
    databank_t1w_dir = Path(databank_t1w_dir)
    traintestset_dir = Path(traintestset_dir)
    
    df_train = pd.read_csv(t1wagepredict_train_csv)
    df_test = pd.read_csv(t1wagepredict_test_csv)
    df_symlink = pd.concat([df_train, df_test], axis=0)
    
    list_job_tuples = []
    
    for i, row in df_symlink.iterrows():
        dataset = row['dataset']
        subject = row['subject']
        session = row['session']
        scan = f"scan-{row['scan']}"
        
        fn_target = databank_t1w_dir / dataset / subject / session / scan / f"{dataset}_{subject}_{session}_{scan}{suffix}"
        fn_link = traintestset_dir / dataset / subject / session / scan / f"{dataset}_{subject}_{session}_{scan}{suffix}"  
        
        list_job_tuples.append((fn_target, fn_link))
    
    return list_job_tuples


def symlink(tuple):
    fn_target, fn_link = tuple
    subprocess.run(['mkdir', '-p', str(Path(fn_link).parent)])
    subprocess.run(['ln', '-sf', fn_target, fn_link])


if __name__ == "__main__":
    databank_t1w_csv = '/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_t1w_v2.csv'
    qa_results_csv = '/nfs/masi/gaoc11/projects/BRAID/data/databank_t1w/quality_assurance/2024-02-05_brain_affine/qa_rater1_freeze.csv'
    braid_train_csv = '/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/braid_train.csv'
    braid_test_csv = '/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/braid_test.csv'
    t1wagepredict_train_csv = '/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/t1wagepredict_train.csv'
    t1wagepredict_test_csv = '/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/t1wagepredict_test.csv'
    databank_t1w_dir = '/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_t1w'
    suffix = '_T1w_brain_MNI152_Warped_crop_downsample.nii.gz'
    traintestset_dir = '/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/T1wAgePredict'

    # df = pd.read_csv(databank_t1w_csv)
    # df = filter_qa_results(df=df, qa_results_csv=qa_results_csv)
    
    # train_test_split_following_braid(
    #     braid_train_csv,
    #     braid_test_csv,
    #     df_filtered = df,
    #     t1wagepredict_train_csv = t1wagepredict_train_csv,
    #     t1wagepredict_test_csv = t1wagepredict_test_csv,
    #     databank_t1w_dir = databank_t1w_dir,
    #     suffix = suffix,
    #     check_existence = True
    # )

    list_job_tuples = create_symlink_job_tuple(t1wagepredict_train_csv, t1wagepredict_test_csv, databank_t1w_dir, suffix, traintestset_dir)
    
    with Pool(processes=8) as pool:
        list(tqdm(pool.imap(symlink, list_job_tuples, chunksize=1), 
                  total=len(list_job_tuples), 
                  desc='symlink'))
