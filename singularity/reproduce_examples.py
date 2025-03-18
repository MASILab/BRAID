import json
import subprocess
from tqdm import tqdm
from pathlib import Path

def create_job_tuples(example_data_root):
    job_tuples = []
    for i in [1,2,3]:
        dir = Path(example_data_root) / f'example_{i}'

        t1 = dir / 'T1w.nii.gz'
        t1_seg = dir / 'T1w_seg.nii.gz'
        dwi = dir / 'dwmri.nii.gz'
        bval = dir / 'dwmri.bval'
        bvec = dir / 'dwmri.bvec'

        with open(dir / 'demog.json') as f:
            demog = json.load(f)
            age = demog['age']
            sex = demog['sex']
            race = demog['race']
        job_tuples.append(
            (str(dwi), str(bval), str(bvec), str(t1), str(t1_seg), age, sex, race, str(dir))
            )
    return job_tuples

def run_braid_in_two_ways(job_tuple):
    dwi, bval, bvec, t1, t1_seg, age, sex, race, dir = job_tuple
    
    # # run braid with python source code
    # outdir = Path(dir) / 'braid_output_python'
    # outdir.mkdir(parents=True, exist_ok=True)
    # log = outdir / 'log.txt'
    # cmd = [
    #     'braid_one_sample_inference', 
    #     '-d', dwi, '-v', bval, '-c', bvec, 
    #     '-t', t1, '-tm', t1_seg,
    #     '-m', 'data/template/MNI_152.nii.gz',
    #     '-a', str(age), '-s', str(sex), '-r', str(race),
    #     '-w', '/home-local/gaoc11/braid_go_public/braid-v1.0', 
    #     '-i', 
    #     '-o', outdir
    # ]
    # with open(log, 'w') as f:
    #     subprocess.run(cmd, stdout=f, stderr=f)
    
    # run braid with singularity
    outdir = Path(dir) / 'braid_output_singularity'
    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / 'log.txt'
    cmd = [
        'singularity', 'run', 
        '-B', f"{dir}:/INPUTS",
        '-B', f"{outdir}:/OUTPUTS",
        '/home-local/gaoc11/braid_go_public/braid_v1.0.0.sif',
    ]
    with open(log, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=f)


if __name__ == "__main__":
    job_tuples = create_job_tuples('singularity/example_data')
    for job_tuple in tqdm(job_tuples, desc="BRAID source code / singularity run"):
        run_braid_in_two_ways(job_tuple)
