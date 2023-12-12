import torch
from torch.utils.data import Dataset
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
import pandas as pd
import numpy as np
import random
from pathlib import Path, PosixPath

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


class BRAID_Dataset(Dataset):
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

        if mode not in ['train', 'test']: raise ValueError("mode must be either 'train' or 'test'")

        if type(subjects) == str:
            if subjects == 'all': 
                self.df = self.df.loc[(self.df['age'] >= self.age_min) & (self.df['age'] <= self.age_max), ]
            else:
                raise ValueError("subjects must be either 'all' or a list/array of subjects")
        else:
            if not '_' in subjects[0]: raise ValueError("subject format must be 'dataset_subject'")
            self.df = self.df.loc[(self.df['age'] >= self.age_min) &
                                  (self.df['age'] <= self.age_max) &
                                  (self.df['dataset_subject'].isin(self.subjects)), ]

        if mode == 'train':
            self.df['sample_weight'] = 1 / self.df.groupby('dataset_subject')['dataset_subject'].transform('count')
            
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            while True:
                age_start = random.choices(range(self.age_min, self.age_max), k=1)[0]
                age_end = age_start + 1
                samples = self.df.loc[(self.df['age'] >= age_start) & (self.df['age'] <= age_end), ]
                if samples.shape[0] >= 1:
                    break
            row = samples.sample(n=1, weights='sample_weight').iloc[0]
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
                
        age = torch.tensor(row['age'].values[0], dtype=torch.float32)

        return images, label_feature, age