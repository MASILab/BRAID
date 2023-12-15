"""
Added new rule of sampling:
- When a scan is sampled, its probability of being sampled (in the future) is reduced.

Author: Chenyu Gao
Date: Dec 14, 2023
"""

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from pathlib import Path, PosixPath
from tqdm import tqdm
import matplotlib.pyplot as plt

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
            self.df['sample_weight'] = self.df['sample_weight'] / self.df['sample_weight'].sum()
            
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
            
            # decay posibility of being sampled, normalize the age bin to sum to 1 (to prevent extreme small values == 0)
            self.df.loc[(self.df['dataset_subject']==row['dataset_subject']) &
                        (self.df['session']==row['session']) &
                        (self.df['scan']==row['scan']), 'sample_weight'] *= 0.01  # smaller factor, faster to cover all samples
            self.df.loc[age_range_mask, 'sample_weight'] = self.df.loc[age_range_mask, 'sample_weight'] / self.df.loc[age_range_mask, 'sample_weight'].sum()

        else:
            row = self.df.iloc[idx]
        
        # unique id for each scan
        scan_id = row['dataset_subject'] + '_' + row['session'] + '_' + str(row['scan'])
        return scan_id, row['age']

# Hyperparameters
batch_size = 4
epochs = 50
milestones = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]

# The entire pool
df = pd.read_csv('/tmp/.GoneAfterReboot/spreadsheet/braid_train.csv')
df = df.loc[(df['age'] >= 45) & (df['age'] < 90), ]
df['scan_id'] = df['dataset_subject'] + '_' + df['session'] + '_' + df['scan'].astype(str)

# Dataloader
dataset_train = BRAID_Dataset(
    dataset_root = '/tmp/.GoneAfterReboot/braid_dataset',
    csv_file = '/tmp/.GoneAfterReboot/spreadsheet/braid_train.csv',
    subjects = 'all',
    age_min = 45,
    age_max = 90,
    mode = 'train',
)
dataloader_train = DataLoader(
    dataset_train, 
    batch_size = batch_size, 
    shuffle = True, 
    num_workers = 0
    )

ages = []
for i in tqdm(range(epochs)):
    for scan_id, age in dataloader_train:
        ages.extend(age)
        df = df[~df['scan_id'].isin(scan_id)]

    if (i+1) in milestones:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5, 10))
        axes[0].hist(ages, bins=range(40, 101), edgecolor='black', color='blue', label='sampled')
        axes[0].set_xlabel('Age')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'After {i+1} epochs\n'
                          f'Ages of scans that have been sampled')
        axes[0].legend(loc='upper right')

        axes[1].hist(df['age'].values, bins=range(40, 101), edgecolor='black', color='red', label='never sampled')
        axes[1].set_xlabel('Age')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Ages of scans that have never been sampled\n'
                          f"{df.shape[0]} scans from {df['dataset_subject'].unique().shape[0]} subjects")
        axes[1].legend(loc='upper right')
        axes[1].set_ylim(0, 400)

        fig.savefig(f'experiments/2023-12-14_Test_DataLoader_reweight/figs/epoch-{i+1}.png', bbox_inches='tight')
        plt.close('all')