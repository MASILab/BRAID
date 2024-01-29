import os
import random
import braid.utls
from tqdm import tqdm
from multiprocessing import Pool

def list_all_preprocessed_t1w(
    databank_root='/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_t1w',
    suffix='_T1w_MNI152_Warped.nii.gz',
    ):
    
    list_t1w = []
    for root, dirs, files in os.walk(databank_root):
        for fn in files:
            if fn.endswith(suffix):
                list_t1w.append(os.path.join(root, fn))
                
    print(f"Found {len(list_t1w)} preprocessed T1w files.")
    return list_t1w
    
    
def get_random_subset(list_t1w, n=200, seed=0):
    
    random.seed(seed)
    list_t1w_sample = random.sample(list_t1w, n)
    
    print(f"Randomly sampled {n} T1w files.")
    return list_t1w_sample


def create_parallel_jobs(
    list_t1w_sample,
    qa_root = '/nfs/masi/gaoc11/projects/BRAID/data/databank_t1w/quality_assurance',
):
    
    list_job_tuples = []
    for t1w in list_t1w_sample:
        path_png = os.path.join(qa_root, t1w.split('/')[-1].replace('.nii.gz', '.png'))
        list_job_tuples.append((t1w, path_png))
    
    print(f"Created {len(list_job_tuples)} jobs.")
    return list_job_tuples


def generate_qa_screenshots(tuple):
    t1w, path_png = tuple
    braid.utls.generate_qa_screenshot_t1w(t1w, path_png, offset=0)


if __name__ == '__main__':
    list_t1w = list_all_preprocessed_t1w()
    list_t1w_sample = get_random_subset(list_t1w)
    list_job_tuples = create_parallel_jobs(list_t1w_sample)
    
    with Pool(processes=8) as pool:
        results = list(tqdm(pool.imap(generate_qa_screenshots, list_job_tuples, chunksize=1), total=len(list_job_tuples), desc='Generating QA screenshots'))