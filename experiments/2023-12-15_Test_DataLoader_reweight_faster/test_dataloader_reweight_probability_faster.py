"""
Problem:
(1) After adding the probability decay feature, the dataloader becomes very slow because 
    the reweighting process is quite expensive,
(2) Previous testing experiments only used num_workers = 0, which hides a big problem away.
    When num_workers > 1, things act differently and unexpectedly in that each worker 
    runs in its own process and does not share memory with other workers. This means that
    our probability decay feature is almost useless because each worker is updating its own 
    copy of the dataframe.

Solution I came up after this testing experiment:
Redesign the dataloader. Split it into two parts: the first part generates a list of lists of scans,
while the second part loops over the list of lists.

Note that the "updated dataloader" in this file is not the solution but rather a curious testing.

Author: Chenyu Gao
Date: Dec 15, 2023
"""

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
from pathlib import Path, PosixPath
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Monitor number of open files (for debugging purpose)
def get_open_file_count():
    pid = os.getpid()
    try:
        proc_fd_dir = f"/proc/{pid}/fd"
        files = os.listdir(proc_fd_dir)
        return len(files)
    except (FileNotFoundError, ProcessLookupError):
        return None
def print_open_file_count():
    open_file_count = get_open_file_count()
    if open_file_count is not None:
        print(f"Current number of open files: {open_file_count}")
    else:
        print("Unable to retrieve open file count.")

def visualize_dist_sampled(ages, df, epoch, prefix='before'):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5, 10))
    axes[0].hist(ages, bins=range(40, 101), edgecolor='black', color='blue', label='sampled')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'After {epoch} epochs\n'
                      f'Ages of scans that have been sampled')
    axes[0].legend(loc='upper right')

    axes[1].hist(df['age'].values, bins=range(40, 101), edgecolor='black', color='red', label='never sampled')
    axes[1].set_xlabel('Age')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Ages of scans that have never been sampled\n'
                      f"{df.shape[0]} scans from {df['dataset_subject'].unique().shape[0]} subjects")
    axes[1].legend(loc='upper right')
    axes[1].set_ylim(0, 400)

    fig.savefig(f'experiments/2023-12-15_Test_DataLoader_reweight_faster/figs/{prefix}_epoch-{epoch}.png', bbox_inches='tight')
    plt.close('all')


# Previous dataloader (simplified version)
class BRAID_Dataset_before(Dataset):
    def __init__(
        self, 
        csv_file: str | PosixPath,
        age_min: int = 45,
        age_max: int = 90,
    ) -> None:
        
        self.df = pd.read_csv(csv_file)
        self.age_min = age_min
        self.age_max = age_max

        self.df = self.df.loc[(self.df['age'] >= self.age_min) & (self.df['age'] < self.age_max), ]
        self.df['scan_id'] = self.df['dataset_subject'] + '_' + self.df['session'] + '_' + 'scan-' + self.df['scan'].astype(str)
        self.df['sample_weight'] = 1 / self.df.groupby('dataset_subject')['dataset_subject'].transform('count')
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        while True:
            age_start = random.choice(range(self.age_min, self.age_max))
            age_end = age_start + 1
            age_range_mask = (self.df['age'] >= age_start) & (self.df['age'] < age_end)

            samples = self.df.loc[age_range_mask, ]
            if samples.shape[0] >= 1:
                break
        row = samples.sample(n=1, weights='sample_weight').iloc[0]
        self.df.loc[self.df['scan_id']==row['scan_id'], 'sample_weight'] *= 0.01  # smaller factor, faster to cover all samples
        self.df.loc[age_range_mask, 'sample_weight'] /= self.df.loc[age_range_mask, 'sample_weight'].sum()

        return row['scan_id'], row['age']

# Updated dataloader
class BRAID_Dataset_after(Dataset):
    def __init__(
        self, 
        csv_file: str | PosixPath,
        age_min: int = 45,
        age_max: int = 90,
    ) -> None:
        
        self.df = pd.read_csv(csv_file)
        self.age_min = age_min
        self.age_max = age_max

        self.df = self.df.loc[(self.df['age'] >= self.age_min) & (self.df['age'] < self.age_max), ]
        self.df['scan_id'] = self.df['dataset_subject'] + '_' + self.df['session'] + '_' + 'scan-' + self.df['scan'].astype(str)
        self.df['sample_weight'] = 1 / self.df.groupby('dataset_subject')['dataset_subject'].transform('count')
            
        self.list_dfs = []
        for age_start in range(self.age_min, self.age_max):
            age_end = age_start + 1
            age_range_mask = (self.df['age'] >= age_start) & (self.df['age'] < age_end)               
            
            samples = self.df.loc[age_range_mask, ]
            if samples.shape[0] >= 1:
                self.list_dfs.append(samples.copy())

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        idx_df = random.choice(range(len(self.list_dfs)))
        row = self.list_dfs[idx_df].sample(n=1, weights='sample_weight').iloc[0]
        self.list_dfs[idx_df].loc[self.list_dfs[idx_df]['scan_id']==row['scan_id'], 'sample_weight'] *= 0.01
        self.list_dfs[idx_df]['sample_weight'] /= self.list_dfs[idx_df]['sample_weight'].sum()

        return row['scan_id'], row['age']


# Test and visualize
batch_size = 8
num_workers = 8
epochs = 10
milestones = [1, 2, 3, 4, 5, 10]


print('----before----')
df = pd.read_csv('/tmp/.GoneAfterReboot/spreadsheet/braid_train.csv')
df = df.loc[(df['age'] >= 45) & (df['age'] < 90), ]
df['scan_id'] = df['dataset_subject'] + '_' + df['session'] + '_' + 'scan-' + df['scan'].astype(str)

dataset_train = BRAID_Dataset_before(
    csv_file = '/tmp/.GoneAfterReboot/spreadsheet/braid_train.csv',
    age_min = 45,
    age_max = 90,
)
dataloader_train = DataLoader(
    dataset_train, 
    batch_size = batch_size, 
    shuffle = True, 
    num_workers = num_workers
    )

ages = []
for i in range(epochs):
    for scan_id, age in tqdm(dataloader_train):
        ages.extend(age.tolist())
        df = df[~df['scan_id'].isin(scan_id)]

    if (i+1) in milestones:
        visualize_dist_sampled(ages, df, epoch=(i+1), prefix='before')


print('----after----')
df = pd.read_csv('/tmp/.GoneAfterReboot/spreadsheet/braid_train.csv')
df = df.loc[(df['age'] >= 45) & (df['age'] < 90), ]
df['scan_id'] = df['dataset_subject'] + '_' + df['session'] + '_' + 'scan-' + df['scan'].astype(str)

dataset_train = BRAID_Dataset_after(
    csv_file = '/tmp/.GoneAfterReboot/spreadsheet/braid_train.csv',
    age_min = 45,
    age_max = 90,
)
dataloader_train = DataLoader(
    dataset_train, 
    batch_size = batch_size, 
    shuffle = True, 
    num_workers = num_workers
    )

ages = []
for i in range(epochs):
    for scan_id, age in tqdm(dataloader_train):
        ages.extend(age.tolist())
        df = df[~df['scan_id'].isin(scan_id)]

    if (i+1) in milestones:
        visualize_dist_sampled(ages, df, epoch=(i+1), prefix='after')
