import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from braid.dataset import get_the_sequence_of_scans, flatten_the_list

def organize_train_val_test_set(
    train_dir, train_csv, list_folds, cv_subjects_dir, num_sample_cycle,
    test_dir, test_csv, tsan_dataset_root, tsan_excel, mapping_csv
    ):

    # Convert to Path object
    train_dir = Path(train_dir)
    cv_subjects_dir = Path(cv_subjects_dir)
    test_dir = Path(test_dir)
    tsan_dataset_root = Path(tsan_dataset_root)

    # Load the csv of unorganized dataset, create dictionary for TSAN excel and mapping csv
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    d_tsan_excel = {'t1w': [], 'age': [], 'sex': []}
    d_mapping = {'original': [], 'tsan': []}

    # Organize the test set
    tsan_img_id = 1
    tsan_dir = tsan_dataset_root / 'test'
    tsan_dir.mkdir(parents=True, exist_ok=True)

    list_scan_id = get_the_sequence_of_scans(
        csv_file=test_csv, subjects='all', age_min=0, age_max=999, mode='test'
    )
    for scan_id in tqdm(list_scan_id, desc='Organizing TSAN test set'):
        dataset = scan_id.split('_')[0]
        subject = scan_id.split('_')[1]
        session = scan_id.split('_')[2]
        scan = scan_id.split('_')[3]
        
        # Create symbolic link and record the mapping
        fn_ori = f"{dataset}_{subject}_{session}_{scan}_T1w_brain_MNI152_Warped_crop_downsample_2mm.nii.gz"
        fn_tsan = "sub-{0:08d}.nii.gz".format(tsan_img_id)
        t1w_ori = test_dir / dataset / subject / session / scan / fn_ori
        t1w_tsan = tsan_dir / fn_tsan
        t1w_tsan.symlink_to(t1w_ori)
        
        d_mapping['original'].append(fn_ori)
        d_mapping['tsan'].append(fn_tsan)
    
        # collect information for TSAN excel
        values = df_test.loc[(df_test['dataset_subject'] == f"{dataset}_{subject}"), 'sex'].values
        values = values[~pd.isna(values)]
        sex = values[0]
        sex_tsan = 1 if sex == 'male' else 0 if sex == 'female' else None
        
        d_tsan_excel['t1w'].append(fn_tsan)
        d_tsan_excel['age'].append(None)
        d_tsan_excel['sex'].append(sex_tsan)

        tsan_img_id += 1

    # Five-fold cross-validation
    for fold in list_folds:

        # load subject splitting of this fold
        try:
            subjects_train = np.load((cv_subjects_dir / f"subjects_fold_{fold}_train.npy"), allow_pickle=True)
            subjects_val = np.load((cv_subjects_dir / f"subjects_fold_{fold}_val.npy"), allow_pickle=True)
        except FileNotFoundError:
            print(f"Fold-{fold} does not have valid subjects splitting.")
            continue

        # Organize the train set
        tsan_dir = tsan_dataset_root / f'fold-{fold}_train'
        tsan_dir.mkdir(parents=True, exist_ok=True)

        list_scan_id = get_the_sequence_of_scans(
            csv_file=train_csv, subjects=subjects_train, age_min=45, age_max=90,
            mode='train', epochs=num_sample_cycle)  # larger the num_sample_cycle, closer to uniform distribution of age
        list_scan_id = flatten_the_list(list_scan_id)

        for scan_id in tqdm(list_scan_id, desc=f'Organizing TSAN train set for fold-{fold}'):
            dataset = scan_id.split('_')[0]
            subject = scan_id.split('_')[1]
            session = scan_id.split('_')[2]
            scan = scan_id.split('_')[3]
            
            # Create symbolic link and record the mapping
            fn_ori = f"{dataset}_{subject}_{session}_{scan}_T1w_brain_MNI152_Warped_crop_downsample_2mm.nii.gz"
            fn_tsan = "sub-{0:08d}.nii.gz".format(tsan_img_id)
            t1w_ori = train_dir / dataset / subject / session / scan / fn_ori
            t1w_tsan = tsan_dir / fn_tsan
            t1w_tsan.symlink_to(t1w_ori)
            
            d_mapping['original'].append(fn_ori)
            d_mapping['tsan'].append(fn_tsan)
        
            # collect information for TSAN excel
            values = df_train.loc[(df_train['dataset_subject'] == f"{dataset}_{subject}"), 'sex'].values
            values = values[~pd.isna(values)]
            sex = values[0]
            sex_tsan = 1 if sex == 'male' else 0 if sex == 'female' else None
            
            values = df_train.loc[(df_train['dataset_subject'] == f"{dataset}_{subject}") &
                                  (df_train['session'] == session), 'age'].values
            values = values[~pd.isna(values)]
            age = values[0]

            d_tsan_excel['t1w'].append(fn_tsan)
            d_tsan_excel['age'].append(age)
            d_tsan_excel['sex'].append(sex_tsan)

            tsan_img_id += 1


        # Organize the val set
        tsan_dir = tsan_dataset_root / f'fold-{fold}_val'
        tsan_dir.mkdir(parents=True, exist_ok=True)

        list_scan_id = get_the_sequence_of_scans(
            csv_file=train_csv, subjects=subjects_val, age_min=45, age_max=90,
            mode='test', epochs=None)
        list_scan_id = flatten_the_list(list_scan_id)
        
        for scan_id in tqdm(list_scan_id, desc=f'Organizing TSAN val set for fold-{fold}'):
            dataset = scan_id.split('_')[0]
            subject = scan_id.split('_')[1]
            session = scan_id.split('_')[2]
            scan = scan_id.split('_')[3]
            
            # Create symbolic link and record the mapping
            fn_ori = f"{dataset}_{subject}_{session}_{scan}_T1w_brain_MNI152_Warped_crop_downsample_2mm.nii.gz"
            fn_tsan = "sub-{0:08d}.nii.gz".format(tsan_img_id)
            t1w_ori = train_dir / dataset / subject / session / scan / fn_ori
            t1w_tsan = tsan_dir / fn_tsan
            t1w_tsan.symlink_to(t1w_ori)
            
            d_mapping['original'].append(fn_ori)
            d_mapping['tsan'].append(fn_tsan)
        
            # collect information for TSAN excel
            values = df_train.loc[(df_train['dataset_subject'] == f"{dataset}_{subject}"), 'sex'].values
            values = values[~pd.isna(values)]
            sex = values[0]
            sex_tsan = 1 if sex == 'male' else 0 if sex == 'female' else None
            
            values = df_train.loc[(df_train['dataset_subject'] == f"{dataset}_{subject}") &
                                  (df_train['session'] == session), 'age'].values
            values = values[~pd.isna(values)]
            age = values[0]

            d_tsan_excel['t1w'].append(fn_tsan)
            d_tsan_excel['age'].append(age)
            d_tsan_excel['sex'].append(sex_tsan)

            tsan_img_id += 1
        
    # Save the TSAN excel and mapping csv
    df = pd.DataFrame(d_tsan_excel)
    Path(tsan_excel).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(tsan_excel, header=False, index=False)
    
    df = pd.DataFrame(d_mapping)
    Path(mapping_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(mapping_csv, index=False)


if __name__ == "__main__":
    
    # Inputs
    train_dir = '/tmp/.GoneAfterReboot/TSAN_dataset/tsan_train'
    train_csv = '/tmp/.GoneAfterReboot/TSAN_dataset/spreadsheet/t1wagepredict_train.csv'
    list_folds = [1, 2, 3, 4, 5]
    cv_subjects_dir = '/tmp/.GoneAfterReboot/TSAN_dataset/cross_validation'
    num_sample_cycle = 3
    test_dir = '/tmp/.GoneAfterReboot/TSAN_dataset/tsan_test'
    test_csv = '/tmp/.GoneAfterReboot/TSAN_dataset/spreadsheet/t1wagepredict_test.csv'
    tsan_dataset_root = '/tmp/.GoneAfterReboot/TSAN_dataset/organized/'
    tsan_excel = '/tmp/.GoneAfterReboot/TSAN_dataset/organized/spreadsheet/dataset.xlsx'
    mapping_csv = '/tmp/.GoneAfterReboot/TSAN_dataset/organized/spreadsheet/mapping.csv'
    
    organize_train_val_test_set(
        train_dir, train_csv, list_folds, cv_subjects_dir, num_sample_cycle,
        test_dir, test_csv,
        tsan_dataset_root, tsan_excel, mapping_csv
        )