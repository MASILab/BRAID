import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from monai.transforms import Compose, LoadImaged, AddChanneld, ToTensord, EnsureChannelFirstd, ResizeD
import pandas as pd
import numpy as np
import random
from pathlib import Path, PosixPath

def vectorize_sex_race(
    sex: str,
    race: str,
    ) -> torch.Tensor:
    
    pass
#TODO: finish this function
    

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
        
        self.dataset_root = dataset_root
        self.df = pd.read_csv(csv_file)
        self.subjects = subjects
        self.age_min = age_min
        self.age_max = age_max
        self.mode = mode
    
        columns_to_check = ['dataset', 'subject', 'session', 'scan', 'sex', 'race_simple', 'age', 'control_label', 'dataset_subject']
        for col in columns_to_check:
            if col not in self.df.columns: raise ValueError(f"Column '{col}' not found in csv file")

        if mode not in ['train', 'test']: raise ValueError("mode must be either 'train' or 'test'")

        if subjects != 'all':
            if not '_' in subjects[0]: raise ValueError("subject format must be 'dataset_subject'")
            self.df = self.df.loc[(self.df['age'] >= self.age_min) &
                                  (self.df['age'] <= self.age_max) &
                                  (self.df['dataset_subject'].isin(self.subjects)), ]
        else:
            self.df = self.df.loc[(self.df['age'] >= self.age_min) &
                                  (self.df['age'] <= self.age_max), ]

        if mode == 'train':
            self.df['sample_weight'] = 1 / self.df.groupby('dataset_subject')['dataset_subject'].transform('count')
            
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            while True:
                age_start = random.choices(range(self.age_min, self.age_max), k=1)[0]
                age_end = age_start + 1
                samples = self.df.loc[(self.df['age']>=age_start) & (self.df['age']<=age_end), ]
                if samples.shape[0] >= 1: break
            row = samples.sample(n=1, weights='sample_weight')
        else:
            row = self.df.iloc[idx]
            
        # TODO: after finishing vectorize_sex_race