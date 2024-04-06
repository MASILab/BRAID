import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


class Visualize_Progression_Data_Points:
    """ Visualize the chronological ages (or intervals before AD/MCI) of the data points 
    of subjects who have progressed from cognitively normal to MCI or AD.
    """
    def __init__(self, pred_csv, databank_csv):
        
        df = pd.read_csv(pred_csv)
        databank = pd.read_csv(databank_csv)
        
        # Retrieve the diagnosis information
        df['diagnosis'] = None
        for i, row in tqdm(df.iterrows(), total=len(df.index), desc='Retrieve diagnosis information'):
            loc_filter = (databank['dataset']==row['dataset']) & (databank['subject']==row['subject']) & ((databank['session']==row['session']) | databank['session'].isnull())
            if row['dataset'] in ['UKBB']:
                control_label = databank.loc[loc_filter, 'control_label'].values[0]
                df.loc[i,'diagnosis'] = 'normal' if control_label == 1 else None
            else:
                df.loc[i,'diagnosis'] = databank.loc[loc_filter, 'diagnosis_simple'].values[0]
        
        self.df = df
        
    def mark_progression_subjects_out(self):
        """ Create the following columns to the dataframe:
            - "age_AD": the age when the subject was diagnosed with AD for the first time.
            - "time_since_AD": the time interval (in years) between the first AD diagnosis and the current time point, 
                        negative means before the diagnosis and vice versa.
            - "age_MCI": the age when the subject was diagnosed with MCI for the first time.
            - "time_since_MCI": the time interval (in years) between the first MCI diagnosis and the current time point, 
                        negative means before the diagnosis and vice versa.
        Note: subjects, whose diagnosis of available sessions begins with MCI or AD, are excluded.
        """
        
        self.df = self.df.loc[self.df['diagnosis'].notna(), ].copy()
        
        for disease in ['AD', 'MCI']:
            self.df[f'age_{disease}'] = None
            
            for subj in self.df.loc[self.df['diagnosis']==disease, 'dataset_subject'].unique():
                rows_subj = self.df.loc[self.df['dataset_subject']==subj, ].copy()
                rows_subj = rows_subj.sort_values(by='age')
                if rows_subj.iloc[0]['diagnosis'] != 'normal':
                    continue
                self.df.loc[self.df['dataset_subject']==subj, f'age_{disease}'] = rows_subj.loc[rows_subj['diagnosis']==disease, 'age'].min()
            self.df[f'time_since_{disease}'] = self.df['age'] - self.df[f'age_{disease}']
            
            num_subj = len(self.df.loc[self.df[f'age_{disease}'].notna(), 'dataset_subject'].unique())
            print(f'Found {num_subj} subjects with {disease} progression.')

    def visualize_data_points(self, png, disease='MCI'):
        assert f"time_since_{disease}" in self.df.columns, f"Column 'time_since_{disease}' is not available."

        df = self.df.loc[self.df[f'age_{disease}'].notna(), ].copy()
        df = df.sort_values(by='age')
        df['y_subject'] = None
        for i, subj in enumerate(df['dataset_subject'].unique()):
            df.loc[df['dataset_subject']==subj, 'y_subject'] = i+1
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 12))
        
        for ax_id, x_axis in enumerate(['age', f'time_since_{disease}']):            
            sns.lineplot(
                data=df,
                x=x_axis, y='y_subject',
                units="dataset_subject",
                estimator=None, 
                lw=1,
                color = 'tab:gray',
                alpha=0.5,
                linestyle='-',
                ax=axes[ax_id]
                )

            sns.scatterplot(
                data=df, 
                x=x_axis, y='y_subject', 
                hue='diagnosis', 
                palette=['tab:green', 'tab:blue', 'tab:orange', 'tab:red'],
                alpha=1,
                ax=axes[ax_id]
                )
            
        fig.savefig(png, dpi=300)
        
        
vis = Visualize_Progression_Data_Points(
    pred_csv='models/2024-02-07_ResNet101_BRAID_warp/predictions/predicted_age_fold-1_test_bc.csv', 
    databank_csv='/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv')
vis.mark_progression_subjects_out()
vis.visualize_data_points('experiments/2024-04-05_MCI_AD_Prediction_From_T_Minus_N_Years/figs/vis_progression_data_points.png', disease='MCI')