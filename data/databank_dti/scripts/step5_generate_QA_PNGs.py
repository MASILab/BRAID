# Generate screenshots of sagittal, coronal, and axial views of FA and MD maps for QA.
# 
# Author: Chenyu Gao
# Date: Dec 1, 2023

import braid.utls
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

def take_screenshots(tuple_job):
    fa, md, png = tuple_job
    try:
        braid.utls.generate_qa_screenshot_fa_md(fa, md, png, offset=0)
    except:
        print(f"failed to generate screenshots:\n{fa}\n{md}\n")

def generate_job_tuples(
    databank_root,
    screenshots_root,
    suffix
):
    print("collecting inputs for all plotting jobs for parallel processing...")
    list_job_tuples = []
    
    for dataset in databank_root.iterdir():
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
                    fa = scan / 'final' / f'fa{suffix}'
                    md = scan / 'final' / f'md{suffix}'
                    if not (fa.exists() and md.exists()):
                        print("\t\tincomplete inputs: {}".format(scan))
                        continue
                    
                    # outputs
                    png = screenshots_root / dataset.name / "{}_{}_{}.png".format(subject.name, session.name, scan.name)
                    if png.is_file():
                        continue

                    list_job_tuples.append((fa, md, png))
    
    return list_job_tuples    
    
    
if __name__ == '__main__':
    databank_root = Path('/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_dti')
    screenshots_root = Path('/home-local/QA_databank_dti')
    suffix = '_skullstrip_MNI152_warped.nii.gz'
    
    list_job_tuples = generate_job_tuples(databank_root, screenshots_root, suffix)
    print(f"{len(list_job_tuples)} PNGs to plot\nAssign them to parallel painters...")

    with Pool(processes=8) as pool:
        list(tqdm(pool.imap(take_screenshots, list_job_tuples, chunksize=1), total=len(list_job_tuples)))