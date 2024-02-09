"""
1. Prepare the dataset in the following format for easy use of the code of TSAN:

    Train Folder-----
            sub-0001.nii.gz
            sub-0002.nii.gz
            .......

    Validation Folder-----
            sub-0003.nii.gz
            sub-0004.nii.gz
            .......
            
    Test Folder-----
            sub-0005.nii.gz
            sub-0006.nii.gz
            .......
            
    Dataset.xls 

    sub-0001.nii.gz     60     1
    sub-0002.nii.gz     74     0
    .......
    
2. We will setup two datapools: one for the training and validation, and the other for the testing.
For the first one, we copy it to local machine. We set up data for 5-fold cross-validation by creating symlinks.
We might need to create duplicated symlinks for the same file to mimic even distribution of age. 
"""
import subprocess
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def setup_TSAN_datapool(
    databank_t1w_dir,
    suffix,
    t1wagepredict_train_csv,
    t1wagepredict_test_csv,
    tsan_train_dir,
    tsan_test_dir,
    ):
        
    databank_t1w_dir = Path(databank_t1w_dir)
    
    config = {
        'dataset': ['train', 'test'],
        'csv': [t1wagepredict_train_csv, t1wagepredict_test_csv],
        'dir': [tsan_train_dir, tsan_test_dir],
    }
    
    for i, dataset in enumerate(config['dataset']): 
        df = pd.read_csv(config['csv'][i])
        tsan_dir = Path(config['dir'][i])
        
        for _, row in tqdm(df.iterrows(), total=len(df.index), desc=f"Setup TSAN {dataset}"):
            dataset = row['dataset']
            subject = row['subject']
            session = row['session']
            scan = f"scan-{row['scan']}"
            
            fn_target = databank_t1w_dir / dataset / subject / session / scan / f"{dataset}_{subject}_{session}_{scan}{suffix}"
            fn_link = tsan_dir / dataset / subject / session / scan / f"{dataset}_{subject}_{session}_{scan}{suffix}"
        
            subprocess.run(['mkdir', '-p', str(Path(fn_link).parent)])
            subprocess.run(['ln', '-sf', fn_target, fn_link])


if __name__ == "__main__":
    databank_t1w_dir = '/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_t1w/'
    suffix = '_T1w_brain_MNI152_Warped_crop_downsample_2mm.nii.gz'
    t1wagepredict_train_csv = '/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/t1wagepredict_train.csv'
    t1wagepredict_test_csv = '/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/t1wagepredict_test.csv'
    tsan_train_dir = '/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/TSAN_dataset/tsan_train'
    tsan_test_dir = '/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/TSAN_dataset/tsan_test'
    
    setup_TSAN_datapool(databank_t1w_dir, suffix, t1wagepredict_train_csv, t1wagepredict_test_csv, tsan_train_dir, tsan_test_dir)