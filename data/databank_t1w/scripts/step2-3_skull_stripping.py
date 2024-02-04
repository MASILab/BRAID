""" Perform skull stripping. If the brain segmentation file exists, use it. Otherwise, use synthstrip.
"""

import os
import pdb
import paramiko
import subprocess
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool


def find_file_on_server(server, dir, fn_endswith):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    ssh.connect(hostname=server)
    command = f"find {dir} \( -type f -o -type l \) -name '*{fn_endswith}'"
    print(f"Executing the command on {server}: {command}")
    stdin, stdout, stderr = ssh.exec_command(command)
    
    list_files = stdout.readlines()
    list_files = [fn.strip() for fn in list_files]
    print(f"Execution finished. Found {len(list_files)} files.")
    
    return list_files


def generate_job_tuples(
    databank_root,
    server='hickory',
):
    list_t1w = find_file_on_server(server=server, dir=databank_root, fn_endswith='_T1w.nii.gz')
    list_t1w_seg = find_file_on_server(server=server, dir=databank_root, fn_endswith='_T1w_seg.nii.gz')
    list_t1w_brain = find_file_on_server(server=server, dir=databank_root, fn_endswith='_T1w_brain.nii.gz')
    
    list_job_tuples = []
    
    for t1w in list_t1w:
        t1w_seg = t1w.replace('_T1w.nii.gz', '_T1w_seg.nii.gz')
        t1w_brain = t1w.replace('_T1w.nii.gz', '_T1w_brain.nii.gz')

        # skip if skull-stripping already done
        if t1w_brain in list_t1w_brain: 
            continue
        
        if t1w_seg in list_t1w_seg:
            list_job_tuples.append((t1w, t1w_seg))
        else:
            list_job_tuples.append((t1w, None))
    
    return list_job_tuples


def skull_strip(tuple):

    # > Manual Settings
    server='hickory'
    tmp_root='/nobackup/p_masi/gaoc11/tmp'
    synthstrip_wrapper='/nobackup/p_masi/gaoc11/synthstrip-singularity'
    # >

    # create tmp dir
    t1w, t1w_seg = tuple
    tmp_dir = Path(tmp_root) / t1w.split('/')[-1].replace('.nii.gz', '')
    subprocess.run(['mkdir', '-p', tmp_dir])
    log = tmp_dir / 'log.txt'

    with open(log, 'a') as f:
        # transfer files to nobackup
        tmp_t1w = tmp_dir / t1w.split('/')[-1]
        subprocess.run(['scp', f"{server}:{t1w}", tmp_t1w], stdout=f, stderr=f)
        if t1w_seg is not None:
            tmp_t1w_seg = tmp_dir / t1w_seg.split('/')[-1]
            subprocess.run(['scp', f"{server}:{t1w_seg}", tmp_t1w_seg], stdout=f, stderr=f)

        # skull strip
        if Path(tmp_t1w).is_file():
            tmp_brain = tmp_dir / t1w.split('/')[-1].replace('_T1w.nii.gz', '_T1w_brain.nii.gz')
            tmp_brain_mask = tmp_dir / t1w.split('/')[-1].replace('_T1w.nii.gz', '_T1w_brain_mask.nii.gz')
            if Path(tmp_t1w_seg).is_file():
                subprocess.run(['fslmaths', tmp_t1w_seg, '-div', tmp_t1w_seg, tmp_brain_mask], stdout=f, stderr=f)
                subprocess.run(['fslmaths', tmp_t1w, '-mul', tmp_brain_mask, tmp_brain], stdout=f, stderr=f)
            else:
                subprocess.run([synthstrip_wrapper, '-i', tmp_t1w, '-o', tmp_brain, '-m', tmp_brain_mask], stdout=f, stderr=f)

            # transfer files back to server
            subprocess.run(['scp', tmp_brain, f"{server}:{t1w.replace('_T1w.nii.gz', '_T1w_brain.nii.gz')}"], stdout=f, stderr=f)
            subprocess.run(['scp', tmp_brain_mask, f"{server}:{t1w.replace('_T1w.nii.gz', '_T1w_brain_mask.nii.gz')}"], stdout=f, stderr=f)

    # remove tmp dir
    subprocess.run(['rm', '-rf', tmp_dir])    

if __name__ == '__main__':
    for dataset in ['ICBM', 'NACC', 'OASIS3', 'OASIS4', 'ROSMAPMARS', 'UKBB', 'VMAP', 'WRAP', 'ADNI', 'BIOCARD', 'BLSA', 'HCPA']:
        list_job_tuples = generate_job_tuples(
            databank_root = f'/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_t1w/{dataset}/',
            server='hickory',
        )

        with Pool(processes=96) as pool:
            list(tqdm(pool.imap(skull_strip, list_job_tuples, chunksize=1), total=len(list_job_tuples), desc=f'skull strip {dataset}'))