""" Crop and downsample the FA, MD images 
and create symlink for the final FA, MD images to use.
"""

import os
import torch
import subprocess
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    CenterSpatialCrop,
    Spacing,
    SaveImage,
)
torch.set_flush_denormal(True)


def find_files(root_dir, suffix):
    
    list_files = []
    for root, dirs, files in os.walk(root_dir):
        for fn in files:
            if fn.endswith(suffix):
                list_files.append(os.path.join(root, fn))
    
    print(f'Found {len(list_files)} files endswith {suffix} under {root_dir}')
    return list_files


class Preprocessing_Dataset(Dataset):
    def __init__(self, list_t1w, data_root_dir):
        
        self.list_t1w = list_t1w
        self.transform = Compose([
            LoadImage(reader="NibabelReader", image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            CenterSpatialCrop(roi_size=(192, 228, 192)),
            Spacing(pixdim=(1.5, 1.5, 1.5), mode='bilinear'), # expected: 128 x 152 x 128, 1.5mm^3
            SaveImage(
                output_postfix='crop_downsample', 
                output_ext='.nii.gz', 
                output_dir=data_root_dir, 
                data_root_dir=data_root_dir,
                separate_folder=False,
                print_log=False,
                ),
        ])
        
    def __len__(self):
        return len(self.list_t1w)
    
    def __getitem__(self, idx):
        self.transform(self.list_t1w[idx])
        return 0


def create_symlink_job_tuple(train_csv, test_csv, databank_dir, suffix, braid_dir):
    
    databank_dir = Path(databank_dir)
    braid_dir = Path(braid_dir)
    
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    df_symlink = pd.concat([df_train, df_test], axis=0)
    
    list_job_tuples = []
    
    for _, row in df_symlink.iterrows():
        dataset = row['dataset']
        subject = row['subject']
        session = row['session']
        scan = f"scan-{row['scan']}"
        
        fa_target = databank_dir / dataset / subject / session / scan / 'final' / f"fa{suffix}"
        md_target = databank_dir / dataset / subject / session / scan / 'final' / f"md{suffix}"
        fa_link = braid_dir / dataset / subject / session / scan / f"fa{suffix}"
        md_link = braid_dir / dataset / subject / session / scan / f"md{suffix}"
        
        if fa_target.is_file() and md_target.is_file():
            list_job_tuples.append((fa_target, fa_link))
            list_job_tuples.append((md_target, md_link))
    
    return list_job_tuples


def symlink(tuple):
    fn_target, fn_link = tuple
    subprocess.run(['mkdir', '-p', str(Path(fn_link).parent)])
    subprocess.run(['ln', '-sf', fn_target, fn_link])


if __name__ == "__main__":
    
    databank_dti = Path('/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_dti')
    suffix = '_skullstrip_MNI152.nii.gz'
    num_workers = 4
    
    # Crop and downsample the FA and MD images
    for dataset_dir in databank_dti.iterdir():
        if not dataset_dir.is_dir():
            continue

        list_t1w = find_files(root_dir=dataset_dir, suffix=suffix)
        dataset = Preprocessing_Dataset(list_t1w=list_t1w, data_root_dir=dataset_dir)
        dataloader = DataLoader(dataset=dataset, batch_size=num_workers, shuffle=False, 
                                num_workers=num_workers, pin_memory = False, prefetch_factor=None)

        for _ in tqdm(dataloader, total=len(dataloader), desc=f'Crop and downsample FA/MD from {dataset_dir.name}'):
            pass

    # Symlink final FA and MD we need to braid_dataset
    braid_dataset = Path('/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/braid_dataset_ss_affine_crop_downsample')
    suffix = '_skullstrip_MNI152_crop_downsample.nii.gz'
    train_csv = '/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/braid_train.csv'
    test_csv = '/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/braid_test.csv'
    
    list_job_tuples = create_symlink_job_tuple(train_csv, test_csv, databank_dti, suffix, braid_dataset)
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(symlink, list_job_tuples, chunksize=1), 
                    total=len(list_job_tuples), 
                    desc='symlink'))