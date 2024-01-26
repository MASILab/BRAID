""" Collect T1w images in the databank_t1w.csv to the target location.
"""

import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def copy_t1w(path_databank_root, row):
    t1w = row['t1w']
    if not Path(t1w).is_file(): 
        return
    elif t1w.endswith('.nii.gz'):
        dataset = row['dataset']
        subject = row['subject']
        session = row['session'] if pd.notnull(row['session']) else 'ses-1'
        scan = 'scan-{}'.format(row['scan'])

        t1w_target = path_databank_root / dataset / subject / session / scan / f'{dataset}_{subject}_{session}_{scan}_T1w.nii.gz'
        
        subprocess.run(['mkdir', '-p', str(t1w_target.parent)])
        subprocess.run(['rsync', '-L', str(t1w), str(t1w_target)])
    else:
        raise ValueError(f'Current script is not expecting suffix other than .nii.gz')


if __name__ == '__main__':
    # target location to save the databank
    path_databank_root = Path('/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_t1w')
    
    # load the csv of the databank
    df = pd.read_csv('/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_t1w.csv')
    df = df.loc[df['age'].notnull(), ]
    
    # copy all t1w to the target location
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc='collecting t1w for databank_t1w'):
        copy_t1w(path_databank_root, row)