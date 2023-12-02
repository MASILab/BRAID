# Generate screenshots of sagittal, coronal, and axial views of FA and MD maps for QA.
# 
# Author: Chenyu Gao
# Date: Dec 1, 2023

import braid.utls
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

def single_job_for_screenshots(tuple_job):
    path_fa, path_md, path_png = tuple_job
    braid.utls.generate_qa_screenshot_fa_md(path_fa, path_md, path_png, offset=0)
    

path_databank_root = Path('/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_dti')
path_screenshots_root = Path('/home-local/QA_databank_dti')

print("collecting inputs for all plotting jobs for parallel processing...")
list_tuple_job = []

for dataset in path_databank_root.iterdir():
    print("\tsearching through dataset: {}".format(dataset.name))
    
    for subject in dataset.iterdir():
        if not (subject.name.startswith('sub-') and subject.is_dir()):
            continue
        
        for session in subject.iterdir():
            if not (session.name.startswith('ses-') and session.is_dir()):
                continue
            
            for scan in session.iterdir():
                if not (scan.name.startswith('scan-') and scan.is_dir()):
                    continue
                
                # inputs
                path_fa = scan / 'final' / 'fa_skullstrip_MNI152.nii.gz'
                path_md = scan / 'final' / 'md_skullstrip_MNI152.nii.gz'
                INPUTS_COMPLETE = path_fa.exists() and path_md.exists()

                # outputs
                path_png = path_screenshots_root / dataset.name / "{}_{}_{}.png".format(subject.name, session.name, scan.name)
                
                if INPUTS_COMPLETE:
                    list_tuple_job.append((path_fa, path_md, path_png))
                else:
                    print("\t\tincomplete inputs: {}".format(scan))

print("Number of PNGs to plot: {}\nAssign them to parallel painters...".format(len(list_tuple_job)))
with Pool(processes=4) as pool:
    results = list(tqdm(pool.imap(single_job_for_screenshots, list_tuple_job, chunksize=1), total=len(list_tuple_job)))