""" Given the predictions directory, draw Bland-Altman plot for each csv.
"""

import re
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

class bland_altman_plot():
    def __init__(
        self, 
        prediction_root,
        databank_csv, 
        crossval_subjects_dir):
        
        # Data
        self.prediction_root = Path(prediction_root)
        self.databank = pd.read_csv(databank_csv)
        self.crossval_subjects_dir = Path(crossval_subjects_dir)
    
        # Plotting parameters
        self.marker_size = 5
        self.marker_linewidth = 0
        self.alpha = {
            'scatter_marker_outside': 0.2,
            'scatter_marker_inside': 0.75,
            'kde': 0.6,
        }
        self.fontfamily = 'DejaVu Sans'
        self.fontsize = {'title': 9, 'label': 11, 'ticks': 11, 'legend': 9}        
        self.xlim = [25, 110]
        self.ylim = [-28, 28]
        self.xticks = [30, 45, 60, 75, 90, 105]
        self.yticks = [-20, -15, -10, -5, 0, 5, 10, 15, 20]

    def plot_testing(self, csv):
        df = pd.read_csv(csv)
        
        # retrieve diagnosis information from databank
        df['diagnosis'] = df.apply(lambda row: self.databank.loc[
            (self.databank['dataset'] == row['dataset']) &
            (self.databank['subject'] == row['subject']) &
            ((self.databank['session'] == row['session']) | (self.databank['session'].isnull())),
            'diagnosis_simple'].values[0], axis=1)
        list_selected_diagnosis = ['normal', 'MCI', 'dementia']
        df = df.loc[df['diagnosis'].isin(list_selected_diagnosis), ]
        
        # prepare data for plotting
        df['age_diff'] = df['age_pred'] - df['age_gt']
        df['testing set'] = np.where(df['dataset'] == 'ICBM', 'external dataset', 'internal dataset')
        
        # plot
        fig, axes = plt.subplots(1, len(list_selected_diagnosis), figsize=(11, 4), sharex=True, sharey=True)

        for i, dx in enumerate(list_selected_diagnosis):
            age_mask = (df['age_gt'] >= 45) & (df['age_gt'] < 90)
            age_mask_inv = ~age_mask
            dx_mask = df['diagnosis'] == dx
            
            sns.scatterplot(
                data=df.loc[age_mask_inv & dx_mask, ],
                x='age_gt',
                y='age_diff',
                s=self.marker_size,
                linewidth=self.marker_linewidth,
                color="tab:gray",
                alpha=self.alpha['scatter_marker_outside'],
                ax=axes[i]
            )
            
            sns.scatterplot(
                data=df.loc[age_mask & dx_mask, ],
                x='age_gt',
                y='age_diff',
                hue='testing set',
                s=self.marker_size,
                linewidth=self.marker_linewidth,
                alpha=self.alpha['scatter_marker_inside'],
                ax=axes[i]
            )

            sns.kdeplot(
                data=df.loc[age_mask & dx_mask, ],
                x='age_gt',
                y='age_diff',
                hue='testing set',
                fill=True,
                levels=10,
                cut=1,
                alpha=self.alpha['kde'],
                ax=axes[i]
            )
            
            axes[i].axhline(y=0, linestyle='-', linewidth=1, color='k', alpha=0.25)
            axes[i].axvline(x=45, linestyle='--', linewidth=1, color='k', alpha=0.25)
            axes[i].axvline(x=90, linestyle='--', linewidth=1, color='k', alpha=0.25, label='age range used for training')
            axes[i].legend(prop={'size': self.fontsize['legend'], 'family': self.fontfamily}, loc='lower left')
            axes[i].set_xlabel('chronological age (years)', fontsize=self.fontsize['label'], fontname=self.fontfamily)
            axes[i].set_ylabel('predicted age - chronological age (years)', fontsize=self.fontsize['label'], fontname=self.fontfamily)
            axes[i].text(
                0.05, 0.95,
                dx, 
                fontsize=self.fontsize['title'],
                fontfamily=self.fontfamily,
                transform=axes[i].transAxes,
                verticalalignment='top', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')
                )
            axes[i].tick_params(axis='both', direction='out', length=2)
            axes[i].set_xlim(self.xlim[0], self.xlim[1])
            axes[i].set_ylim(self.ylim[0], self.ylim[1])
            axes[i].set_xticks(self.xticks)
            axes[i].set_yticks(self.yticks)
        
        # Save figure
        save_png = str(csv).replace('/predictions/', '/figs/bland_altman_v1/').replace('.csv', '.png')
        Path(save_png).parent.mkdir(parents=True, exist_ok=True)

        fig.subplots_adjust(hspace=0,wspace=0.1)
        fig.savefig(save_png,
                    dpi=300,
                    bbox_inches='tight')
        
    def plot_training(self, csv):
        df = pd.read_csv(csv)
        
        # retrive train/val splitting information
        if 'set' not in df.columns:
            match = re.search(r'fold-(\d+)', csv.name)
            fold_idx = int(match.group(1))

            df['set'] = None
            subj_train = np.load((self.crossval_subjects_dir / f"subjects_fold_{fold_idx}_train.npy"), allow_pickle=True)
            subj_val = np.load((self.crossval_subjects_dir / f"subjects_fold_{fold_idx}_val.npy"), allow_pickle=True)
            df.loc[df['dataset_subject'].isin(subj_train), 'set'] = 'train'
            df.loc[df['dataset_subject'].isin(subj_val), 'set'] = 'val'
        
        # plot
        df['age_diff'] = df['age_pred'] - df['age_gt']
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        age_mask = (df['age_gt'] >= 45) & (df['age_gt'] < 90)
        age_mask_inv = ~age_mask
            
        sns.scatterplot(
            data=df.loc[age_mask_inv, ],
            x='age_gt',
            y='age_diff',
            s=self.marker_size,
            linewidth=self.marker_linewidth,
            color="tab:gray",
            alpha=self.alpha['scatter_marker_outside'],
            ax=ax
        )

        sns.scatterplot(
            data=df.loc[age_mask, ],
            x='age_gt',
            y='age_diff',
            hue='set',
            hue_order=['val', 'train'],
            s=self.marker_size,
            linewidth=self.marker_linewidth,
            alpha=self.alpha['scatter_marker_inside'],
            ax=ax
        )
            
        ax.axhline(y=0, linestyle='-', linewidth=1, color='k', alpha=0.25)
        ax.axvline(x=45, linestyle='--', linewidth=1, color='k', alpha=0.25)
        ax.axvline(x=90, linestyle='--', linewidth=1, color='k', alpha=0.25, label='age range used for training')
        ax.legend(prop={'size': self.fontsize['legend'], 'family': self.fontfamily}, loc='lower left')
        ax.set_xlabel('chronological age (years)', fontsize=self.fontsize['label'], fontname=self.fontfamily)
        ax.set_ylabel('predicted age - chronological age (years)', fontsize=self.fontsize['label'], fontname=self.fontfamily)
        ax.text(
            0.05, 0.95,
            'normal',
            fontsize=self.fontsize['title'],
            fontfamily=self.fontfamily,
            transform=ax.transAxes,
            verticalalignment='top', 
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')
            )
        ax.tick_params(axis='both', direction='out', length=2)
        ax.set_xlim(self.xlim[0], self.xlim[1])
        ax.set_ylim(self.ylim[0], self.ylim[1])
        ax.set_xticks(self.xticks)
        ax.set_yticks(self.yticks)
        
        # Save figure
        save_png = str(csv).replace('/predictions/', '/figs/bland_altman_v1/').replace('.csv', '.png')
        Path(save_png).parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(save_png,
                    dpi=300,
                    bbox_inches='tight')
            
    def plot_for_every_prediction_csv(self):
        csvs = sorted(self.prediction_root.glob('predicted_age_fold-*.csv'))
        
        for csv in tqdm(csvs, total=len(csvs), desc='Bland-Altman Plot'):
            
            if 'test' in csv.name:
                self.plot_testing(csv)
                
            elif 'trainval' in csv.name:
                self.plot_training(csv)
                
            else:
                print(f'csv name {csv.name} is not classified correctly.')        
    
if __name__ == "__main__":
    selected_models = [
        {
            'prediction_root': 'models/2024-02-07_ResNet101_BRAID_warp/predictions',
            'databank_csv': '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv'
        },
        {
            'prediction_root': 'models/2024-02-13_TSAN_second_stage/predictions',
            'databank_csv': '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_t1w_v2.csv'
        },
        {
            'prediction_root': 'models/2024-02-07_T1wAge_ResNet101/predictions',
            'databank_csv': '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_t1w_v2.csv'
        
        },
        {
            'prediction_root': 'models/2023-12-22_ResNet101/predictions',
            'databank_csv': '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv'
        },
        {
            'prediction_root': 'models/2024-01-16_ResNet101_MLP/predictions',
            'databank_csv': '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv'
        },
    ]
    crossval_subjects_dir = '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/cross_validation/'
    
    for m in selected_models:
        b = bland_altman_plot(
            prediction_root = m['prediction_root'],
            databank_csv = m['databank_csv'],
            crossval_subjects_dir = crossval_subjects_dir,
        )
        b.plot_for_every_prediction_csv()