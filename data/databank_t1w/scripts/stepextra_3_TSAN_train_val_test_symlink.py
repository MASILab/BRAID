import subprocess
import pandas as pd
from pathlib import Path
from braid.dataset import get_the_sequence_of_scans

def organize_test_set(
    test_dir,
    test_csv,
    tsan_test_dir,
    tsan_test_excel,
    mapping_csv,
):
    test_dir = Path(test_dir)
    tsan_test_dir = Path(tsan_test_dir)
    if not tsan_test_dir.is_dir():
        subprocess.run(['mkdir', '-p', str(tsan_test_dir)])
    
    df = pd.read_csv(test_csv)
    
    d_tsan_test_excel = {'t1w': [], 'age': [], 'sex': []}
    d_mapping_csv = {'original': [], 'tsan': []}
    
    list_scan_id = get_the_sequence_of_scans(
        csv_file = test_csv, subjects = 'all', 
        age_min = 0, age_max = 999, mode = 'test')
    
    for i, scan_id in enumerate(list_scan_id):
        dataset = scan_id.split('_')[0]
        subject = scan_id.split('_')[1]
        session = scan_id.split('_')[2]
        scan = scan_id.split('_')[3]
        
        # Create symbolic link and record the mapping
        fn_ori = f"{dataset}_{subject}_{session}_{scan}_T1w_brain_MNI152_Warped_crop_downsample_2mm.nii.gz"
        fn_tsan = "sub-{0:08d}.nii.gz".format(i+1)
        t1w_ori = test_dir / dataset / subject / session / scan / fn_ori
        t1w_tsan = tsan_test_dir / fn_tsan
        
        subprocess.run(['ln', '-sf', t1w_ori, t1w_tsan])
        print(t1w_tsan)
        
        d_mapping_csv['original'].append(fn_ori)
        d_mapping_csv['tsan'].append(fn_tsan)
    
        # collect information for TSAN excel
        values = df.loc[(df['dataset_subject'] == f"{dataset}_{subject}"), 'sex'].values
        values = values[~pd.isna(values)]
        sex = values[0]
        sex_tsan = 1 if sex == 'male' else 0 if sex == 'female' else None
        
        d_tsan_test_excel['t1w'].append(fn_tsan)
        d_tsan_test_excel['age'].append(None)
        d_tsan_test_excel['sex'].append(sex_tsan)

    df = pd.DataFrame(d_tsan_test_excel)
    if not Path(tsan_test_excel).parent.is_dir():
        subprocess.run(['mkdir', '-p', str(Path(tsan_test_excel).parent)])
    df.to_excel(tsan_test_excel, header=False, index=False)
    
    df = pd.DataFrame(d_mapping_csv)
    if not Path(mapping_csv).parent.is_dir():
        subprocess.run(['mkdir', '-p', str(Path(mapping_csv).parent)])
    df.to_csv(mapping_csv, index=False)


if __name__ == "__main__":
    local_tsan_dataset_root = '/tmp/.GoneAfterReboot/TSAN_dataset/'
    local_tsan_organized_root = '/tmp/.GoneAfterReboot/TSAN_dataset/organized/'
    num_sample_cycle = 3  # The larger the number, the closer to uniform distribution of age, but at the cost of heavier disk usage and computation.
    
    
    test_dir = '/tmp/.GoneAfterReboot/TSAN_dataset/tsan_test'
    test_csv = '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/t1wagepredict_test.csv'
    tsan_test_dir = '/tmp/.GoneAfterReboot/TSAN_dataset/organized/testset/'
    tsan_test_excel = '/tmp/.GoneAfterReboot/TSAN_dataset/organized/spreadsheet/testset.xlsx'
    mapping_csv = '/tmp/.GoneAfterReboot/TSAN_dataset/organized/spreadsheet/testset_mapping.csv'
    
    organize_test_set(
        test_dir,
        test_csv,
        tsan_test_dir,
        tsan_test_excel,
        mapping_csv,
    )