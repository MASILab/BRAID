""" register t1w to MNI152
designed for running on the landman01 cluster

Author: Chenyu Gao
Date: 2024-01-26
"""

import paramiko
import subprocess
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool


def find_file_on_server(server, dir, fn_endswith):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    ssh.connect(hostname=server)
    command = f"find {dir} -type f -name '*{fn_endswith}'"
    print(f"Executing the command on {server}: {command}")
    stdin, stdout, stderr = ssh.exec_command(command)
    
    list_files = stdout.readlines()
    list_files = [fn.strip() for fn in list_files]
    print("Execution finished.")
    
    return list_files


def create_parallel_jobs(path_databank_t1w, server='hickory'):
    """Create a list of tuples, each tuple contains job inputs for parallel processing.
    
    """
    list_t1w = find_file_on_server(server=server, dir=path_databank_t1w, fn_endswith='_T1w.nii.gz')
    list_t1w_mni = find_file_on_server(server=server, dir=path_databank_t1w, fn_endswith='_T1w_MNI152_Warped.nii.gz')
    
    list_t1w_to_be_registered = []
    
    for t1w in tqdm(list_t1w, total=len(list_t1w), desc='Creating registration jobs'):
        if t1w.replace('_T1w.nii.gz', '_T1w_MNI152_Warped.nii.gz') in list_t1w_mni:
            continue
        else:
            list_t1w_to_be_registered.append(t1w)
            
    print(f"Number of jobs created: {len(list_t1w_to_be_registered)}")
    return list_t1w_to_be_registered
        
    
def register_t1w_to_MNI(path_t1w):
    
    tmp_root = '/home/gaoc11/tmp'
    
    log = f"{tmp_root}/{path_t1w.split('/')[-1].replace('.nii.gz', '')}.log"
    with open(log, 'a') as f:
        
        tmp_folder = Path(tmp_root) / path_t1w.split('/')[-1].replace('.nii.gz', '')
        subprocess.run(['mkdir', '-p', str(tmp_folder)])
        print(f'Created temporary folder: {tmp_folder}', file=f)
    
        tmp_t1w = tmp_folder / path_t1w.split('/')[-1]
        subprocess.run(['scp', f"hickory:{path_t1w}", str(tmp_t1w)], stdout=f, stderr=f)
        
        # ANTs registration
        tmp_out_prefix = str(tmp_t1w).replace('_T1w.nii.gz', '_T1w_MNI152_')
        tmp_out = str(tmp_t1w).replace('_T1w.nii.gz', '_T1w_MNI152_Warped.nii.gz')
        tmp_out_affine_mat = str(tmp_t1w).replace('_T1w.nii.gz', '_T1w_MNI152_0GenericAffine.mat')
    
        command = ['antsRegistrationSyN.sh', '-d', '3', 
                '-f', '/nfs2/ForChenyu/MNI_152.nii.gz', 
                '-m', str(tmp_t1w), 
                '-o', tmp_out_prefix, 
                '-t', 'a']
        subprocess.run(command, stdout=f, stderr=f)
    
        # scp files to server
        subprocess.run(['scp', tmp_out, f"hickory:{Path(path_t1w).parent}"], stdout=f, stderr=f)
        subprocess.run(['scp', tmp_out_affine_mat, f"hickory:{Path(path_t1w).parent}"], stdout=f, stderr=f)

    # remove temporary files    
    subprocess.run(['rm', '-r', str(tmp_folder)])
    

if __name__ == '__main__':
    list_t1w_to_be_registered = create_parallel_jobs('/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_t1w', server='hickory')
    
    with Pool(processes=96) as pool:
        results = list(tqdm(pool.imap(register_t1w_to_MNI, list_t1w_to_be_registered, chunksize=1), total=len(list_t1w_to_be_registered)))