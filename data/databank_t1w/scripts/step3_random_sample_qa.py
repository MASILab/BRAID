""" Randomly sample preprocessed T1w images and generate QA screenshots.
Prepare a .csv file listing all selected samples.
"""

import os
import random
import subprocess
import braid.utls
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

def list_preprocessed_t1w(data_root, suffix):
    list_t1w = []
    for root, dirs, files in os.walk(data_root):
        for fn in files:
            if fn.endswith(suffix):
                list_t1w.append(os.path.join(root, fn))
                
    print(f"Found {len(list_t1w)} preprocessed T1w in {data_root}")
    return list_t1w
    
    
def get_random_subset(list_t1w, n=500, seed=0):
    n = len(list_t1w) if n > len(list_t1w) else n

    random.seed(seed)
    list_t1w_sample = random.sample(list_t1w, n)
    return list_t1w_sample


def create_jobs(databank_root, suffix, n, seed, qa_root, qa_csv):
    qa_root = Path(qa_root)
    if not qa_root.is_dir():
        subprocess.run(['mkdir', '-p', str(qa_root)])
    path_qa_csv = qa_root / qa_csv

    list_job_tuples = []
    list_df_t1w = []

    for dataset in Path(databank_root).iterdir():
        if not dataset.is_dir():
            continue
        candidates = list_preprocessed_t1w(data_root=dataset, suffix=suffix)
        list_t1w_sample = get_random_subset(candidates, n=n, seed=seed)
        list_df_t1w.extend(list_t1w_sample)
        
        for t1w in list_t1w_sample:
            path_png = qa_root / dataset.name / t1w.split('/')[-1].replace('.nii.gz', '.png')
            list_job_tuples.append((t1w, path_png))
    
    # save the selected samples to csv
    d = {'t1w': list_df_t1w}
    df = pd.DataFrame(data=d)
    df['qa'] = None
    df.to_csv(path_qa_csv, index=False)
    
    return list_job_tuples


def generate_qa_screenshots(tuple):
    t1w, path_png = tuple
    braid.utls.generate_qa_screenshot_t1w(t1w, path_png, offset=0)


if __name__ == '__main__':
    list_job_tuples = create_jobs(
        databank_root = '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_t1w', 
        suffix = '_T1w_brain_MNI152_Warped.nii.gz', 
        n = 500, 
        seed = 0, 
        qa_root = '/nfs/masi/gaoc11/projects/BRAID/data/databank_t1w/quality_assurance/2024-02-05_brain_affine', 
        qa_csv = 'qa.csv')
    
    with Pool(processes=8) as pool:
        list(tqdm(pool.imap(generate_qa_screenshots, list_job_tuples, chunksize=1), 
                  total=len(list_job_tuples), 
                  desc='Generating QA screenshots'))