""" Apply the affine transformation to the skull-stripped T1w image to MNI space.
run on hickory
"""

import os
import subprocess
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

def apply_affine_transformation(tuple):
    input, output, affine = tuple
    
    if not Path(output).parent.is_dir():
        subprocess.run(['mkdir', '-p', str(Path(output).parent)])
    
    command = [
        'antsApplyTransforms', '-d', '3',
        '-i', str(input),
        '-r', '/nfs2/ForChenyu/MNI_152.nii.gz',
        '-o', str(output),
        '-n', 'Linear',
        '-t', str(affine)
    ]
    subprocess.run(command)


def generate_transformation_jobs(databank_root):
    list_job_tuples = []
    
    list_inputs = []
    for root, dirs, files in os.walk(databank_root):
        for fn in files:
            if fn.endswith('_T1w_brain.nii.gz'):
                list_inputs.append(os.path.join(root, fn))

    for input in list_inputs:
        output = input.replace('_T1w_brain.nii.gz', '_T1w_brain_MNI152_Warped.nii.gz')
        affine = input.replace('_T1w_brain.nii.gz', '_T1w_MNI152_0GenericAffine.mat')
        if Path(affine).is_file() and not Path(output).is_file():
            list_job_tuples.append((input, output, affine))
    
    return list_job_tuples


if __name__ == '__main__':
    list_job_tuples = generate_transformation_jobs(
        databank_root='/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_t1w')
    with Pool(processes=8) as pool:
        list(tqdm(pool.imap(apply_affine_transformation, list_job_tuples, chunksize=1), 
                  total=len(list_job_tuples), 
                  desc='t1w_brain -> MNI152 (affine)'))
