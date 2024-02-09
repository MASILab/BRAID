""" Crop, downsample T1w images for TSAN.
"""

import os
import torch
from tqdm import tqdm
from pathlib import Path
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

def list_t1w_mni152(root_dir, suffix):
    
    list_t1w = []
    for root, dirs, files in os.walk(root_dir):
        for fn in files:
            if fn.endswith(suffix):
                list_t1w.append(os.path.join(root, fn))
    
    print(f'Found {len(list_t1w)} T1w images in MNI152 space for {root_dir}')            
    return list_t1w


class Preprocessing_Dataset(Dataset):
    def __init__(self, list_t1w, data_root_dir):
        
        self.list_t1w = list_t1w
        self.transform = Compose([
            LoadImage(reader="NibabelReader", image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=(2.0, 2.0, 2.0), mode='bilinear'),  # Found that spacing will try to output even-number dimension like (92, 110, 92).
            CenterSpatialCrop(roi_size=(91, 109, 91)),         # So we swap the CenterSpatialCrop and Spacing
            SaveImage(
                output_postfix='crop_downsample_2mm', 
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
    databank_t1w = Path('/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_t1w')
    list_datasets = [fd for fd in databank_t1w.iterdir() if fd.is_dir()]
    
    for dataset_dir in list_datasets:
        if dataset_dir.name in []:
            print(f'Manually skip {dataset_dir.name}')
            continue

        list_t1w = list_t1w_mni152(
            root_dir=dataset_dir, 
            suffix='_T1w_brain_MNI152_Warped.nii.gz'
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
            pin_memory = False,
            prefetch_factor = 2,
        )

        for _ in tqdm(dataloader, total=len(dataloader), desc='Crop, downsample T1w for TSAN'):
            pass
