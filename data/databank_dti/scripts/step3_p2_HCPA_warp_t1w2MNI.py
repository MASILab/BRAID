""" Perform non-linear registration of the T1w images to the MNI template using ANTs.
Run on cluster in parallel.
Date: Feb 1, 2024
"""

import os
import pdb
import paramiko
import subprocess
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

def find_file_on_server(server, dir, filename):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    ssh.connect(hostname=server)
    command = f"find {dir} \( -type f -o -type l \) -name '{filename}'"
    print(f"Executing the command on {server}: {command}")
    stdin, stdout, stderr = ssh.exec_command(command)
    
    list_files = stdout.readlines()
    list_files = [fn.strip() for fn in list_files]
    print(f"Execution finished. Found {len(list_files)} files.")
    
    return list_files

def generate_job_tuples(path_dataset, tmp_root, mni_template='/nobackup/p_masi/gaoc11/MNI_152.nii.gz'):
    """ Return a list of tuples, each containing the following:
        - t1w: t1w (GDPR)
        - tmp_t1w: t1w (tmp folder on cluster)
        - mni_template: MNI template (nobackup on cluster)
        - tmp_transform_prefix: prefix of outputs (in tmp)
        - tmp_transform_affine: affine matrix from t1w to template (in tmp)
        - tmp_transform_warp: warp matrix from t1w to template (in tmp)
        - transform_affine: affine matrix (GDPR)
        - transform_warp (GDPR)
    """
    
    list_t1w = find_file_on_server(server='hickory', dir=path_dataset, filename='t1w.nii.gz')
    list_transform_warp = find_file_on_server(server='hickory', dir=path_dataset, filename='transform_t1toMNI_warp.nii.gz')

    list_job_tuples = []

    for t1w in list_t1w:
        transform_warp = t1w.replace('/t1w/', '/transform/').replace('/t1w.nii.gz', '/transform_t1toMNI_warp.nii.gz')
        # skip if the transformation matrix is already there
        if transform_warp in list_transform_warp:
            continue
        
        tmp_folder = Path(tmp_root) / t1w.replace(path_dataset, '').replace('/','').replace('t1w.nii.gz', '')  # to avoid conflicting
        tmp_t1w = tmp_folder / 't1w.nii.gz'
        tmp_transform_prefix = tmp_folder / 'transform_t1toMNI152_'
        tmp_transform_affine = tmp_folder / 'transform_t1toMNI152_0GenericAffine.mat'
        tmp_transform_warp = tmp_folder / 'transform_t1toMNI152_1Warp.nii.gz'
        
        transform_affine = t1w.replace('/t1w/', '/transform/').replace('/t1w.nii.gz', '/transform_t1toMNI_affine.mat')
        
        list_job_tuples.append((t1w, tmp_t1w, mni_template, tmp_transform_prefix, 
                                tmp_transform_affine, tmp_transform_warp,
                                transform_affine, transform_warp))
        
    return list_job_tuples


def setup_inputs(tuple):
    """Seperate this process out to avoid intense I/O operations in parallel.
    """
    t1w, tmp_t1w, mni_template, tmp_transform_prefix, tmp_transform_affine, tmp_transform_warp, transform_affine, transform_warp = tuple
    
    subprocess.run(['mkdir', '-p', str(Path(tmp_t1w).parent)])

    log = Path(tmp_t1w).parent / 'log.txt'
    with open(log, 'a') as f:
        subprocess.run(['scp', f"hickory:{t1w}", str(tmp_t1w)], stdout=f, stderr=f)


def registration_job(tuple):
    t1w, tmp_t1w, mni_template, tmp_transform_prefix, tmp_transform_affine, tmp_transform_warp, transform_affine, transform_warp = tuple
    
    log = Path(tmp_t1w).parent / 'log.txt'
    with open(log, 'a') as f:
        command = [
            'antsRegistrationSyN.sh', '-d', '3',
            '-f', mni_template,
            '-m', str(tmp_t1w),
            '-o', str(tmp_transform_prefix),
            '-t', 's'
            ] 
        subprocess.run(command, stdout=f, stderr=f)


def transfer_output_and_freespace(tuple):
    t1w, tmp_t1w, mni_template, tmp_transform_prefix, tmp_transform_affine, tmp_transform_warp, transform_affine, transform_warp = tuple
    
    # Transfer outputs to server
    subprocess.run(['scp', str(tmp_transform_affine), f"hickory:{transform_affine}"])
    subprocess.run(['scp', str(tmp_transform_warp), f"hickory:{transform_warp}"])
    
    # Remove temporary files
    subprocess.run(['rm', '-rf', str(Path(tmp_t1w).parent)])   


if __name__ == '__main__':
    list_job_tuples = generate_job_tuples(
        path_dataset='/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_dti/HCPA/', 
        tmp_root='/nobackup/p_masi/gaoc11/tmp', 
        mni_template='/nobackup/p_masi/gaoc11/MNI_152.nii.gz'
    )

    with Pool(processes=2) as pool:
        list(tqdm(pool.imap(setup_inputs, list_job_tuples, chunksize=1), total=len(list_job_tuples), desc='Setup inputs'))
    with Pool(processes=96) as pool:
        list(tqdm(pool.imap(registration_job, list_job_tuples, chunksize=1), total=len(list_job_tuples), desc='Registration SyN'))
    with Pool(processes=1) as pool:
        list(tqdm(pool.imap(transfer_output_and_freespace, list_job_tuples, chunksize=1), total=len(list_job_tuples), desc='Transfer outputs and free space'))