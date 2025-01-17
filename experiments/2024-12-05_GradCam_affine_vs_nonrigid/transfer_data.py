# Execute on hickory to transfer testing data of 
# WM age nonrigid and WM age affine to local disk.
# 
# Steps:
# 1. Prepare list of files to transfer.
# 2. tar.gz the files.
# 3. Transfer the tar.gz to local disk.
# 
# Date: Dec 9, 2024

import os
import subprocess
import pandas as pd
from pathlib import Path


dict_dataset_type_suffix = {
    'braid_dataset_ss_affine_crop_downsample': '_skullstrip_MNI152_crop_downsample.nii.gz',
    'braid_dataset_ss_affine_warp_crop_downsample': '_skullstrip_MNI152_warped_crop_downsample.nii.gz',
}


if __name__ == '__main__':
    
    if not os.path.ismount('/home/gaoc11/GDPR'):
        raise Exception("GDPR is not mounted.")
    
    root = "/home/gaoc11/GDPR/masi/gaoc11/BRAID/data"
    os.chdir(root)
    root = Path(root)
    df = pd.read_csv('./dataset_splitting/spreadsheet/braid_test.csv')
    
    # 1. Prepare list of files to transfer.
    paths_txt = 'transfer_testing.txt'
    with open(paths_txt, 'w') as f:
        for dataset_type in dict_dataset_type_suffix.keys():
            for _, row in df.iterrows():
                
                for image_type in ['fa', 'md']:
                    img = root / dataset_type / row['dataset'] / row['subject'] / row['session'] / f"scan-{row['scan']}" / f"{image_type}{dict_dataset_type_suffix[dataset_type]}"
                    if img.readlink().exists():
                        f.write(f"{img.relative_to(root)}\n")
                    else:
                        print(f"Broken symlink: {img}")
    print("List of files to transfer is prepared.")
    
    # 2. tar.gz the files.
    fn_targz = 'braid_testing.tar.gz'
    cmd = ['tar', '-czf', fn_targz, '--dereference', '--files-from', paths_txt]
    subprocess.run(cmd)
    print("tar.gz is created.")
    
    # 3. Transfer the tar.gz to local disk.
    address = 'masi-49.vuds.vanderbilt.edu'
    cmd = ['scp', fn_targz, f'{address}:/tmp/.GoneAfterReboot/{fn_targz}']
    subprocess.run(cmd)
    print("tar.gz is transferred.")
    