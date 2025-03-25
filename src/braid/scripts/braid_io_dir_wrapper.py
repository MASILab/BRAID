import json
import argparse
import subprocess
from pathlib import Path


def check_input_complete(input_dir):
    input_dir = Path(input_dir)
    inputs = [
        'demog.json',
        'dwmri.bval', 
        'dwmri.bvec', 
        'dwmri.nii.gz', 
        'T1w_seg.nii.gz', 
        'T1w.nii.gz'
    ]
    for input in inputs:
        if not (input_dir / input).exists():
            print(f'{input} not found in {input_dir}')
            return False
        
    with open(input_dir / 'demog.json', 'r') as f:
        demog = json.load(f)
    for k in ['age', 'sex', 'race']:
        if k not in demog:
            print(f'{k} not found in demog.json. If it is not available, set it to null in json.')
            return False
        
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir',
                        type=str, 
                        default='/INPUTS',
                        help='path to the input directory.')
    parser.add_argument('-m', '--mni152',
                        type=str, 
                        default='/FILES/MNI_152.nii.gz',
                        help='path to the NIFTI file of the MNI152 template image.')
    parser.add_argument('-w', '--weights',
                        type=str,
                        default='/FILES/braid-v1.0',
                        help='path to the directory of model weights pulled from Hugging Face.')
    parser.add_argument('-o', '--output_dir', 
                        type=str, 
                        default='/OUTPUTS', 
                        help='path to the output directory.')
    args = parser.parse_args()
    
    if not check_input_complete(args.input_dir):
        print("Incomplete input.")
        return
    
    input_dir = Path(args.input_dir)
    t1 = input_dir / 'T1w.nii.gz'
    t1_seg = input_dir / 'T1w_seg.nii.gz'
    dwi = input_dir / 'dwmri.nii.gz'
    bval = input_dir / 'dwmri.bval'
    bvec = input_dir / 'dwmri.bvec'

    with open(input_dir / 'demog.json') as f:
        demog = json.load(f)
        age = demog.get('age', None)
        sex = demog.get('sex', None)
        race = demog.get('race', None)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = output_dir / 'log.txt'
    cmd = [
        'braid_one_sample_inference', 
        '-d', dwi, '-v', bval, '-c', bvec, 
        '-t', t1, '-tm', t1_seg,
        '-m', args.mni152,
        '-w', args.weights,
        '-i', 
        '-o', output_dir
    ]

    if age is not None:
        cmd.extend(['-a', str(age)])
    if sex is not None:
        cmd.extend(['-s', str(sex)])
    if race is not None:
        cmd.extend(['-r', str(race)])

    with open(log, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=f)
    
    return