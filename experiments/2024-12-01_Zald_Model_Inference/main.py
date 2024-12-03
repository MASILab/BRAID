import subprocess
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool


def create_job_tuples(dataset_root):
    job_tuples = []
    
    dataset_root = Path(dataset_root)
    for session in dataset_root.glob('./sub-*/ses-*'):
        anat_folder = session / 'anat'
        dwi_folder =  session / 'dwi'

        t1 = list(anat_folder.glob('*_T1w.nii.gz'))[0]
        t1_seg = anat_folder / 'SLANT_TICV.nii.gz'
        dwi = dwi_folder / 'dwmri.nii.gz'
        bval = dwi_folder / 'dwmri.bval'
        bvec = dwi_folder / 'dwmri.bvec'

        if not t1.exists() or not t1_seg.exists() or not dwi.exists() or not bval.exists() or not bvec.exists():
            print(f"Skipping {session} due to missing files")
            continue
        job_tuples.append((str(dwi), str(bval), str(bvec), str(t1), str(t1_seg), str(session / 'braid')))
    
    print(f"Total instances: {len(job_tuples)}")

    return job_tuples


def preprocessing(job_tuple):
    dwi, bval, bvec, t1, t1_seg, outdir = job_tuple
    log = Path(outdir) / 'log.txt'
    log.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'braid_one_sample_inference', '-d', dwi, '-v', bval, '-c', bvec, '-t', t1, '-tm', t1_seg,
        '-m', '/nfs/masi/gaoc11/projects/BRAID/data/template/MNI_152.nii.gz',
        '-w', '/home-local/gaoc11/braid_go_public/braid-v1.0', '-i', '--preprocess_only',
        '-o', outdir
    ]

    with open(log, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=f)


def complete_inference(job_tuple):
    dwi, bval, bvec, t1, t1_seg, outdir = job_tuple
    log = Path(outdir) / 'log_complete.txt'
    log.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'braid_one_sample_inference', '-d', dwi, '-v', bval, '-c', bvec, '-t', t1, '-tm', t1_seg,
        '-m', '/nfs/masi/gaoc11/projects/BRAID/data/template/MNI_152.nii.gz',
        '-w', '/home-local/gaoc11/braid_go_public/braid-v1.0', '-i',
        '-o', outdir
    ]
    
    with open(log, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=f)
        

def combine_all_csv(dataset_root, save_csv):
    dataset_root = Path(dataset_root)
    all_csv = list(dataset_root.glob('sub-*/ses-*/braid/final/braid_predictions.csv'))
    
    all_dfs = []
    for path_csv in all_csv:
        df = pd.read_csv(path_csv)
        all_dfs.append(df)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(save_csv, index=False)

def edit_columns(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df['subject_id'] = df['path_dwi'].str.split('/').str.get(-4)
    df.drop(['path_dwi','path_t1w'], axis=1, inplace=True)
    
    cols = ['subject_id'] + [col for col in df.columns if col != 'subject_id']
    df = df[cols]

    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    dataset_root = '/fs5/p_masi/rudravg/GF_project/derivaties/WM_Age'
    # job_tuples = create_job_tuples(dataset_root=dataset_root)

    # with Pool(processes=16) as pool:
    #     list(tqdm(pool.imap(preprocessing, job_tuples, chunksize=1), total=len(job_tuples), desc='BRAID preprocessing'))
        
    # for job_tuple in tqdm(job_tuples, total=len(job_tuples), desc='BRAID Inference'):
    #     complete_inference(job_tuple)

    combine_all_csv(dataset_root=dataset_root, save_csv='experiments/2024-12-01_Zald_Model_Inference/zald_braid_inference_2024-12-02.csv')
    edit_columns(
        input_csv='experiments/2024-12-01_Zald_Model_Inference/zald_braid_inference_2024-12-02.csv', 
        output_csv='experiments/2024-12-01_Zald_Model_Inference/zald_braid_inference_2024-12-02_delivery.csv')
    