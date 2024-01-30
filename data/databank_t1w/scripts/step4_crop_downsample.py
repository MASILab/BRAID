""" Crop (no interpolation) and downsample (with interpolation) the T1w images in MNI152.
"""

import os
import torch
from tqdm import tqdm
from pathlib import Path, PosixPath
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

def list_t1w_mni152(root_dir, suffix='_T1w_MNI152_Warped.nii.gz'):
    
    list_t1w = []
    for root, dirs, files in os.walk(root_dir):
        for fn in files:
            if fn.endswith(suffix):
                list_t1w.append(os.path.join(root, fn))
    
    print(f'Found {len(list_t1w)} T1w images in MNI152 space for {root_dir}')            
    return list_t1w


class Preprocessing_Dataset(Dataset):
    def __init__(
        self,
        list_t1w: list[str] | list[PosixPath],
        data_root_dir: str | PosixPath,
    ) -> None:
        
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


if __name__ == "__main__":
    databank_t1w = Path('/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_t1w')
    list_datasets = [fd for fd in databank_t1w.iterdir() if fd.is_dir()]
    
    for dataset_dir in list_datasets:
        if dataset_dir.name in ['UKBB']:
            print(f'Manually skip {dataset_dir.name}')
            continue

        list_t1w = list_t1w_mni152(
            root_dir=dataset_dir, 
            suffix='_T1w_MNI152_Warped.nii.gz'
            )
        
        dataset = Preprocessing_Dataset(
            list_t1w=list_t1w,
            data_root_dir=dataset_dir,
            )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size = 4,
            shuffle = False,
            num_workers = 8,
            pin_memory = True,
            prefetch_factor = 2,
        )

        list(tqdm(dataloader, total=len(dataloader), desc='Crop and downsample T1w in MNI152'))
