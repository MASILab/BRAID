""" Collect T1w images and existing segmentations in the databank_t1w.csv to the target location.
Run on hickory.
"""

import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

def collect(tuple):
    t1w, t1w_target, t1w_seg, t1w_seg_target = tuple
    subprocess.run(['mkdir', '-p', str(t1w_target.parent)])

    try:
        subprocess.run(['ln', '-s', str(t1w), str(t1w_target)])
    except:
        print(f'Error: {t1w_target} -> {t1w}')
    
    try:
        subprocess.run(['ln', '-s', str(t1w_seg), str(t1w_seg_target)])
    except:
        print(f'Error: {t1w_seg_target} -> {t1w_seg}')
        
        
def generate_collector_tuples(
    databank_root,
    databank_csv,
):
    """Generate list of tuples of inputs for parallel collecting.

    Args:
        databank_root (str): target location to save the databank
        databank_csv (str): csv file that contains the information of the databank
    """
    databank_root = Path(databank_root)
    
    df = pd.read_csv(databank_csv)
    df = df.loc[df['age'].notnull(), ]
    
    list_job_tuples = []
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc='generate collector tuples'):
        t1w = row['t1w']
        t1w_seg = row['brainseg']
        
        if not Path(t1w).is_file():
            continue
        
        elif t1w.endswith('.nii.gz'):
            dataset = row['dataset']
            subject = row['subject']
            session = row['session'] if pd.notnull(row['session']) else 'ses-1'
            scan = 'scan-{}'.format(row['scan'])
            
            t1w_target = databank_root / dataset / subject / session / scan / f'{dataset}_{subject}_{session}_{scan}_T1w.nii.gz'
            t1w_seg_target = databank_root / dataset / subject / session / scan / f'{dataset}_{subject}_{session}_{scan}_T1w_seg.nii.gz'
            
            list_job_tuples.append(
                (t1w, t1w_target, t1w_seg, t1w_seg_target)
            )
        else:
            print(f'Current script is not expecting suffix other than .nii.gz')
    
    return list_job_tuples


if __name__ == '__main__':
    list_job_tuples  = generate_collector_tuples(
        databank_root='/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_t1w',
        databank_csv='/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_t1w_v2.csv'
        )
    with Pool(processes=4) as pool:
        list(tqdm(pool.imap(collect, list_job_tuples, chunksize=1), total=len(list_job_tuples), desc='Initialize databank_t1w'))