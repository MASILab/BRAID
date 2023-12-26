import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path, PosixPath
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd,
    Orientationd, 
    CenterSpatialCropd,
    Spacingd, 
    ToTensord,
    ConcatItemsd,
)


def vectorize_sex_race(
    sex: str,
    race: str,
    ) -> torch.Tensor:

    sex2vec = {
        'female': [1, 0],
        'male': [0, 1],
        'unknown': [0.5, 0.5],
    }

    race2vec = {
        'white': [1, 0, 0, 0], 
        'asian': [0, 1, 0, 0], 
        'black or african american': [0, 0, 1, 0],
        'American Indian or Alaska Native': [0, 0, 0, 1],
        'some other race': [0.25, 0.25, 0.25, 0.25],
        'unknown': [0.25, 0.25, 0.25, 0.25],
    }
    
    try:
        sex = sex.lower()
    except:
        sex = 'unknown'
    try:
        race = race.lower()
    except:
        race = 'unknown'

    if sex in sex2vec.keys():
        vec_sex = sex2vec[sex]
    else:
        vec_sex = sex2vec['unknown']
    vec_sex = torch.tensor(vec_sex, dtype=torch.float32)
    
    if race in race2vec.keys():
        vec_race = race2vec[race]
    else:
        vec_race = race2vec['unknown']
    vec_race = torch.tensor(vec_race, dtype=torch.float32)

    return torch.cat((vec_sex, vec_race), dim=0)

# This version is correct only when num_workers = 0
class BRAID_Dataset_Single_Worker(Dataset):
    def __init__(
        self, 
        dataset_root: str | PosixPath,
        csv_file: str | PosixPath,
        subjects: np.ndarray | list[str] | str = 'all',
        age_min: int = 45,
        age_max: int = 90,
        mode: str = 'train',
    ) -> None:
        
        self.dataset_root = Path(dataset_root)
        self.df = pd.read_csv(csv_file)
        self.subjects = subjects
        self.age_min = age_min
        self.age_max = age_max
        self.mode = mode
    
        columns_to_check = ['dataset', 'subject', 'session', 'scan', 'sex', 'race_simple', 'age', 'control_label', 'dataset_subject']
        for col in columns_to_check:
            if col not in self.df.columns: raise ValueError(f"Column '{col}' not found in csv file")
        self.df['scan_id'] = self.df['dataset_subject'] + '_' + self.df['session'] + '_' + 'scan-' + self.df['scan'].astype(str)

        if mode not in ['train', 'test']: raise ValueError("mode must be either 'train' or 'test'")

        if type(subjects) == str:
            if subjects == 'all': 
                self.df = self.df.loc[(self.df['age'] >= self.age_min) & (self.df['age'] < self.age_max), ]
            else:
                raise ValueError("subjects must be either 'all' or a list/array of subjects")
        else:
            if not '_' in subjects[0]: raise ValueError("subject format must be 'dataset_subject'")
            self.df = self.df.loc[(self.df['age'] >= self.age_min) &
                                  (self.df['age'] < self.age_max) &
                                  (self.df['dataset_subject'].isin(self.subjects)), ]

        if mode == 'train':
            self.df['sample_weight'] = 1 / self.df.groupby('dataset_subject')['dataset_subject'].transform('count')
            
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            while True:
                age_start = random.choice(range(self.age_min, self.age_max))
                age_end = age_start + 1
                age_range_mask = (self.df['age'] >= age_start) & (self.df['age'] < age_end)

                samples = self.df.loc[age_range_mask, ]
                if samples.shape[0] >= 1:
                    break
            row = samples.sample(n=1, weights='sample_weight').iloc[0]
            # decay probability (Normalization for preventing vanishing weights)
            self.df.loc[self.df['scan_id']==row['scan_id'], 'sample_weight'] *= 0.01  # smaller the factor, faster it takes to cover all samples
            self.df.loc[age_range_mask, 'sample_weight'] = self.df.loc[age_range_mask, 'sample_weight'] / self.df.loc[age_range_mask, 'sample_weight'].sum()

        else:
            row = self.df.iloc[idx]
        
        fa = self.dataset_root / row['dataset'] / row['subject'] / row['session'] / f"scan-{row['scan']}" / 'fa_skullstrip_MNI152.nii.gz'        
        md = self.dataset_root / row['dataset'] / row['subject'] / row['session'] / f"scan-{row['scan']}" / 'md_skullstrip_MNI152.nii.gz'
        data_dict = {'fa': fa, 'md': md}
        transform = Compose([
            LoadImaged(keys=['fa', 'md'], image_only=False),
            EnsureChannelFirstd(keys=['fa', 'md']),
            Orientationd(keys=['fa', 'md'], axcodes="RAS"),
            CenterSpatialCropd(keys=['fa', 'md'], roi_size=(192, 228, 192)),
            Spacingd(keys=['fa', 'md'], pixdim=(1.5, 1.5, 1.5), mode=('bilinear', 'bilinear')),  # expected: 128 x 152 x 128, 1.5mm^3
            ToTensord(keys=['fa', 'md']),
            ConcatItemsd(keys=['fa', 'md'], name='images')  
        ])
        data_dict = transform(data_dict)
        images = data_dict['images']
        
        sex = row['sex']
        race = row['race_simple']
        label_feature = vectorize_sex_race(sex, race)
                
        age = torch.tensor(row['age'], dtype=torch.float32)

        return images, label_feature, age, row['scan_id']


def get_the_sequence_of_scans(
    csv_file: str | PosixPath,
    subjects: np.ndarray | list[str] | str = 'all',
    age_min: int = 45,
    age_max: int = 90,
    mode: str = 'train',
    epochs: int = 1,
    decay_factor: float = 0.01,
) -> list[str] | list[list[str]]:
    
    # load and check the dataframe
    df = pd.read_csv(csv_file)
    columns_to_check = ['dataset', 'subject', 'session', 'scan', 'sex', 'race_simple', 'age', 'control_label', 'dataset_subject']
    for col in columns_to_check:
        if col not in df.columns: raise ValueError(f"Column '{col}' not found in csv file")
    
    # filter the dataframe by age and subjects
    age_mask = (df['age'] >= age_min) & (df['age'] < age_max)
    if type(subjects) == str:
        if subjects == 'all': 
            df = df.loc[age_mask, ]
        else:
            raise ValueError("subjects must be either 'all' or a list/array of subjects")
    else:
        if not '_' in subjects[0]: raise ValueError("subject format must be '{dataset}_{subject_id}'")
        df = df.loc[age_mask & df['dataset_subject'].isin(subjects), ]
    
    df['scan_id'] = df['dataset_subject'] + '_' + df['session'] + '_' + 'scan-' + df['scan'].astype(str)
    
    if mode == 'train':
        print('Generating the sequence of scans (with uniformly distributed ages) for training...')
        list_scans = []
        df['sample_weight'] = 1 / df.groupby('dataset_subject')['dataset_subject'].transform('count')
        for epoch in tqdm(range(epochs)):
            list_scans_epoch = []
            for _ in range(df.shape[0]):
                while True:
                    age_start = random.choice(range(age_min, age_max))
                    age_end = age_start + 1
                    age_range_mask = (df['age'] >= age_start) & (df['age'] < age_end)
                    samples = df.loc[age_range_mask, ]
                    if samples.shape[0] >= 1:
                        break
                row = samples.sample(n=1, weights='sample_weight').iloc[0]
                
                # probability decay
                df.loc[df['scan_id']==row['scan_id'], 'sample_weight'] *= decay_factor
                df.loc[age_range_mask, 'sample_weight'] = df.loc[age_range_mask, 'sample_weight'] / df.loc[age_range_mask, 'sample_weight'].sum()

                list_scans_epoch.append(row['scan_id'])
            list_scans.append(list_scans_epoch)
    elif mode == 'test':
        print('Generating the sequence of scans for testing...')
        list_scans = df['scan_id'].tolist()

    else:
        raise ValueError("mode must be either 'train' or 'test'")
    
    return list_scans


def flatten_the_list(lst):
    flattened_list = []
    for item in lst:
        if isinstance(item, list):
            flattened_list.extend(flatten_the_list(item))
        else:
            flattened_list.append(item)
    return flattened_list


class BRAID_Dataset(Dataset):
    def __init__(
        self,
        dataset_root: str | PosixPath,
        csv_file: str | PosixPath,
        list_scans: list[str] | list[list[str]],
    ) -> None:
        
        self.dataset_root = Path(dataset_root)
        self.df = pd.read_csv(csv_file)
        self.df['scan_id'] = self.df['dataset_subject'] + '_' + self.df['session'] + '_' + 'scan-' + self.df['scan'].astype(str)
        self.list_scans = flatten_the_list(list_scans)
        
        self.transform = Compose([
            LoadImaged(keys=['fa', 'md'], image_only=False),
            EnsureChannelFirstd(keys=['fa', 'md']),
            Orientationd(keys=['fa', 'md'], axcodes="RAS"),
            CenterSpatialCropd(keys=['fa', 'md'], roi_size=(192, 228, 192)),
            Spacingd(keys=['fa', 'md'], pixdim=(1.5, 1.5, 1.5), mode=('bilinear', 'bilinear')),  # expected: 128 x 152 x 128, 1.5mm^3
            ToTensord(keys=['fa', 'md']),
            ConcatItemsd(keys=['fa', 'md'], name='images')  
        ])
        
    def __len__(self):
        return len(self.list_scans)
    
    def __getitem__(self, idx):
        row = self.df.loc[self.df['scan_id']==self.list_scans[idx], ].iloc[0]

        fa = self.dataset_root / row['dataset'] / row['subject'] / row['session'] / f"scan-{row['scan']}" / 'fa_skullstrip_MNI152.nii.gz'
        md = self.dataset_root / row['dataset'] / row['subject'] / row['session'] / f"scan-{row['scan']}" / 'md_skullstrip_MNI152.nii.gz'
        data_dict = {'fa': fa, 'md': md}
        data_dict = self.transform(data_dict)
        images = data_dict['images']
        
        label_feature = vectorize_sex_race(row['sex'], row['race_simple'])
        age = torch.tensor(row['age'], dtype=torch.float32)

        return images, label_feature, age


def get_BRAID_dataloader(
    dataset_root: str | PosixPath,
    csv_file: str | PosixPath,
    list_scans: list[str] | list[list[str]],    
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 8,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    ) -> DataLoader:

    dataset = BRAID_Dataset(dataset_root, csv_file, list_scans)
    dataloader = DataLoader(
        dataset, 
        batch_size = batch_size, 
        shuffle = shuffle, 
        num_workers = num_workers,
        pin_memory = pin_memory,
        prefetch_factor = prefetch_factor,
        )
    
    return dataloader