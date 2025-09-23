import pdb
import socket
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def setup_dataset(outdir):
    assert socket.gethostname() == 'hickory.accre.vanderbilt.edu', "This script must be run on hickory to access GDPR."

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
        
    # Merge BRAID's T1w train and test csv
    train_csv = Path('/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/t1wagepredict_train.csv')
    test_csv = Path('/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/t1wagepredict_test.csv')
    assert train_csv.exists() and test_csv.exists(), "files not found."
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    # Exclude samples from UKBB
    df = df.loc[df['dataset'] != 'UKBB', :].reset_index(drop=True)
    
    # Symlink the T1w files to outdir
    path_databank_t1w = Path('/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_t1w')
    assert path_databank_t1w.is_dir(), "databank_t1w directory not found."
    for i, row in tqdm(df.iterrows(), total=len(df.index), desc='Symlink T1w files'):
        src = path_databank_t1w / row['dataset'] / row['subject'] / row['session'] / f"scan-{row['scan']}" / f"{row['dataset']}_{row['subject']}_{row['session']}_scan-{row['scan']}_T1w_brain_MNI152_Warped_crop_downsample.nii.gz"
        if not src.exists():
            print(f"File not found: {src}")
            continue
        
        dst = outdir / src.relative_to(path_databank_t1w)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(src)
    
    # Retrieve diagnosis label from databank csv
    databank_csv = Path('/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_t1w_v2.csv')
    assert databank_csv.exists(), "databank_t1w_v2.csv not found."
    databank = pd.read_csv(databank_csv)
    df['diagnosis'] = None
    for i, row in tqdm(df.iterrows(), total=len(df.index), desc='Retrieve diagnosis label'):
        filter = (databank['dataset']==row['dataset']) & (databank['subject']==row['subject']) & ((databank['session']==row['session']) | databank['session'].isnull())
        df.loc[i,'diagnosis'] = databank.loc[filter, 'diagnosis_simple'].values[0]
    
    # Split data into train(80%)/ val(10%)/ test(10%) for age estimation task
    df['age_est_split'] = None
    subjects_with_disease = df.loc[df['diagnosis'] != 'normal', 'dataset_subject'].unique().tolist()  # 1551
    subjects_cn = [ds for ds in df['dataset_subject'].unique() if ds not in subjects_with_disease]  # 4124
    np.random.seed(42)
    np.random.shuffle(subjects_cn)
    train_end = int(len(subjects_cn) * 0.8)
    val_end = int(len(subjects_cn) * 0.9)
    subjects_train = subjects_cn[:train_end]
    subjects_val = subjects_cn[train_end:val_end]
    subjects_test = subjects_cn[val_end:]
    
    df.loc[df['dataset_subject'].isin(subjects_train), 'age_est_split'] = 'train'
    df.loc[df['dataset_subject'].isin(subjects_val), 'age_est_split'] = 'val'
    df.loc[df['dataset_subject'].isin(subjects_test), 'age_est_split'] = 'test'
    
    # Remove redundant columns
    df['race'] = df['race_simple']
    cols_keep = ['dataset','subject','session','scan','sex','race','age','diagnosis','dataset_subject','age_est_split']
    df = df[cols_keep]
    df.to_csv(outdir/'t1w_data_table.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, required=True, help='Output directory for saving the dataset.')
    args = parser.parse_args()
    setup_dataset(args.outdir)