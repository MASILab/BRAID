""" We ran model inference on both the training (train + val) and the testing set for each fold.
Now we need to organize the TSAN outputs for easier comparison with BRAID predictions.

For each fold, there will be two output csv files:
    - predictions/predicted_age_fold-*_trainval.csv
    - predictions/predicted_age_fold-*_test.csv

They share common columns:
    dataset,subject,session,scan,sex,race_simple,age,control_label,dataset_subject,age_gt,age_pred

predicted_age_fold-*_trainval.csv will contain an extra column "set" with value "train" or "val".
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# dataframe of the T1wAgePredict dataset
df_test = pd.read_csv('/tmp/.GoneAfterReboot/TSAN_dataset/spreadsheet/t1wagepredict_test.csv')
df_train = pd.read_csv('/tmp/.GoneAfterReboot/TSAN_dataset/spreadsheet/t1wagepredict_train.csv')
cv_subjects_dir = Path('/tmp/.GoneAfterReboot/TSAN_dataset/cross_validation')
basic_cols = [
    'dataset','subject','session','scan','sex',
    'race_simple','age','control_label','dataset_subject'
    ]

# TSAN outdir and mapping
model_root = 'models/2024-02-13_TSAN_second_stage/weights'
df_mapping = pd.read_csv('models/2024-02-13_TSAN_second_stage/mapping.csv')
pred_root = Path('models/2024-02-13_TSAN_second_stage/predictions')

for fold in Path(model_root).iterdir():
    print(f"Organizing {fold.name}")
    fold_idx = fold.name.split('-')[-1]

    # TSAN predictions in raw
    pred_raw_train = np.load((fold / 'brain_age_train.npz'))
    pred_raw_val = np.load((fold / 'brain_age_val.npz'))
    pred_raw_test = np.load((fold / 'brain_age_test.npz'))

    # _trainval.csv
    df_pred_train = df_train[basic_cols].copy()
    df_pred_train['age_gt'] = df_pred_train['age']  # redundant column for consistency with BRAID output
    df_pred_train['age_pred'] = np.nan
    
    df_pred_train['set'] = None
    subjects_train = np.load((cv_subjects_dir / f"subjects_fold_{fold_idx}_train.npy"), allow_pickle=True)
    subjects_val = np.load((cv_subjects_dir / f"subjects_fold_{fold_idx}_val.npy"), allow_pickle=True)
    df_pred_train.loc[df_pred_train['dataset_subject'].isin(subjects_train), 'set'] = 'train'
    df_pred_train.loc[df_pred_train['dataset_subject'].isin(subjects_val), 'set'] = 'val'
    
    # assign tsan predictions to the _trainval.csv
    for prediction, id in tqdm(zip(pred_raw_train['prediction'], pred_raw_train['ID']), desc='train'):
        scan_id = df_mapping.loc[df_mapping['tsan'] == id, 'original'].unique().item()
        dataset = scan_id.split('_')[0]
        subject = scan_id.split('_')[1]
        session = scan_id.split('_')[2]
        scan = int(scan_id.split('_')[3].replace('scan-', ''))
            
        set = df_pred_train.loc[(df_pred_train['dataset'] == dataset) &
                                (df_pred_train['subject'] == subject), 'set'].unique().item()
        assert set == 'train'
        
        df_pred_train.loc[(df_pred_train['dataset'] == dataset) &
                          (df_pred_train['subject'] == subject) &
                          (df_pred_train['session'] == session) &
                          (df_pred_train['scan'] == scan), 'age_pred'] = prediction
    
    for prediction, id in tqdm(zip(pred_raw_val['prediction'], pred_raw_val['ID']), desc='val'):
        scan_id = df_mapping.loc[df_mapping['tsan'] == id, 'original'].unique().item()
        dataset = scan_id.split('_')[0]
        subject = scan_id.split('_')[1]
        session = scan_id.split('_')[2]
        scan = int(scan_id.split('_')[3].replace('scan-', ''))

        set = df_pred_train.loc[(df_pred_train['dataset'] == dataset) &
                                (df_pred_train['subject'] == subject), 'set'].unique().item()
        assert set == 'val'
        
        df_pred_train.loc[(df_pred_train['dataset'] == dataset) &
                          (df_pred_train['subject'] == subject) &
                          (df_pred_train['session'] == session) &
                          (df_pred_train['scan'] == scan), 'age_pred'] = prediction
    # _test.csv
    df_pred_test = df_test[basic_cols].copy()
    df_pred_test['age_gt'] = df_pred_test['age'] # redundant column for consistency
    df_pred_test['age_pred'] = np.nan
    
    # assign tsan predictions to the _test.csv
    for prediction, id in tqdm(zip(pred_raw_test['prediction'],pred_raw_test['ID']), desc='test'):
        scan_id = df_mapping.loc[df_mapping['tsan'] == id, 'original'].unique().item()
        dataset = scan_id.split('_')[0]
        subject = scan_id.split('_')[1]
        session = scan_id.split('_')[2]
        scan = int(scan_id.split('_')[3].replace('scan-', ''))
        
        df_pred_test.loc[(df_pred_test['dataset'] == dataset) &
                         (df_pred_test['subject'] == subject) &
                         (df_pred_test['session'] == session) &
                         (df_pred_test['scan'] == scan), 'age_pred'] = prediction
    
    # save the .csv
    df_pred_train.to_csv(pred_root / f'predicted_age_fold-{fold_idx}_trainval.csv', index=False)
    df_pred_test.to_csv(pred_root / f'predicted_age_fold-{fold_idx}_test.csv', index=False)
