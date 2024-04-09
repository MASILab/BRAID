import pdb
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def roster_brain_age_models():
    dict_models = {
        'WM age (purified)': {
            'prediction_root': 'models/2024-02-07_ResNet101_BRAID_warp/predictions',
            'col_suffix': '_wm_age_purified',
        },
        'GM age (ours)': {
            'prediction_root': 'models/2024-02-07_T1wAge_ResNet101/predictions',
            'col_suffix': '_gm_age_ours',
        },
        'WM age (contaminated)': {
            'prediction_root': 'models/2023-12-22_ResNet101/predictions',
            'col_suffix': '_wm_age_contaminated',
        },
        'GM age (TSAN)': {
            'prediction_root': 'models/2024-02-12_TSAN_first_stage/predictions',
            'col_suffix': '_gm_age_tsan',
        },
        # 'GM age (DeepBrainNet)': {
        #     'prediction_root': 'models/2024-04-04_DeepBrainNet/predictions',
        #     'col_suffix': '_gm_age_dbn',
        # },        
    }
    return dict_models

class DataPreparation:
    def __init__(self, dict_models, databank_csv):
        self.dict_models = dict_models
        self.databank = pd.read_csv(databank_csv)
    
    def load_predictions_of_all_models(self, bias_correction=True):
        """ Load dataframes from each fold (if cross-validation was used) of each model 
        and combine them into a single wide dataframe.
        If bias_correction is False, load the results before bias correction.
        """
        
        bc = '_bc' if bias_correction else ''
        
        for i, model in enumerate(self.dict_models.keys()):
            pred_root = Path(self.dict_models[model]['prediction_root'])
            col_suffix = self.dict_models[model]['col_suffix']
            
            if model == 'GM age (DeepBrainNet)':
                pred_csv = pred_root / f'predicted_age_test{bc}.csv'
                df_model = pd.read_csv(pred_csv)
                df_model = df_model.groupby(['dataset','subject','session','sex','age'])['age_pred'].mean().reset_index()
                df_model = df_model.rename(columns={'age_pred': f'age_pred{col_suffix}'})
            else:
                for fold_idx in tqdm([1,2,3,4,5], desc=f'Loading data for {model}'):
                    pred_csv = pred_root / f"predicted_age_fold-{fold_idx}_test{bc}.csv"
                    if fold_idx == 1:
                        df_model = pd.read_csv(pred_csv)
                        df_model = df_model.groupby(['dataset','subject','session','sex','age'])['age_pred'].mean().reset_index()
                        df_model = df_model.rename(columns={'age_pred': f'age_pred{col_suffix}_{fold_idx}'})
                    else:
                        tmp = pd.read_csv(pred_csv)
                        tmp = tmp.groupby(['dataset','subject','session','sex','age'])['age_pred'].mean().reset_index()
                        tmp = tmp.rename(columns={'age_pred': f'age_pred{col_suffix}_{fold_idx}'})
                        df_model = df_model.merge(tmp, on=['dataset','subject','session','sex','age'])
            
            if i == 0:
                df = df_model.copy()
            else:
                df = df.merge(df_model.copy(), on=['dataset','subject','session','sex','age'])
        
        print(f"Predictions loaded. DataFrame shape: {df.shape}")
        return df
    
    def retrieve_diagnosis_label(self, df):
        """ Retrieve diagnosis information from the databank and add as a new column "diagnosis".
        """
        df['diagnosis'] = None
        
        for i, row in tqdm(df.iterrows(), total=len(df.index), desc='Retrieve diagnosis information'):
            loc_filter = (self.databank['dataset']==row['dataset']) & (self.databank['subject']==row['subject']) & ((self.databank['session']==row['session']) | self.databank['session'].isnull())
            if row['dataset'] in ['UKBB']:
                control_label = self.databank.loc[loc_filter, 'control_label'].values[0]
                df.loc[i,'diagnosis'] = 'normal' if control_label == 1 else None
            else:
                df.loc[i,'diagnosis'] = self.databank.loc[loc_filter, 'diagnosis_simple'].values[0]
        
        df['diagnosis'] = df['diagnosis'].replace('dementia', 'AD')
        return df
    
    def mark_progression_subjects_out(self, df):
        """ Create the following columns to the dataframe:
            - "age_AD": the age when the subject was diagnosed with AD for the first time.
            - "time_since_AD": the time interval (in years) between the first AD diagnosis and the current time point, 
                        negative means before the diagnosis and vice versa.
            - "age_MCI": the age when the subject was diagnosed with MCI for the first time.
            - "time_since_MCI": the time interval (in years) between the first MCI diagnosis and the current time point, 
                        negative means before the diagnosis and vice versa.
        Note: subjects, whose diagnosis of available sessions begins with MCI or AD, are excluded.
        """
        
        df = df.loc[df['diagnosis'].isin(['normal', 'MCI', 'AD']), ].copy()
    
        if 'dsubj' not in df.columns:
            df['dsubj'] = df['dataset'] + '_' + df['subject']
        
        for disease in ['AD', 'MCI']:
            df[f'age_{disease}'] = None
            
            for subj in df.loc[df['diagnosis']==disease, 'dsubj'].unique():
                rows_subj = df.loc[df['dsubj']==subj, ].copy()
                rows_subj = rows_subj.sort_values(by='age')
                if rows_subj.iloc[0]['diagnosis'] != 'normal':
                    continue
                df.loc[df['dsubj']==subj, f'age_{disease}'] = rows_subj.loc[rows_subj['diagnosis']==disease, 'age'].min()
            df[f'time_since_{disease}'] = df['age'] - df[f'age_{disease}']
            
            num_subj = len(df.loc[df[f'age_{disease}'].notna(), 'dsubj'].unique())
            print(f'Found {num_subj} subjects with {disease} progression.')
        
        return df

    def visualize_data_points(self, df, png, disease='MCI'):
        """ Visualize the chronological ages (or intervals before AD/MCI) of the data points 
        of subjects who have progressed from cognitively normal to MCI or AD.
        """

        assert f"time_since_{disease}" in df.columns, f"Column 'time_since_{disease}' is not available."
        
        df = df.loc[df[f'age_{disease}'].notna(), ].copy()
        if 'dsubj' not in df.columns:
            df['dsubj'] = df['dataset'] + '_' + df['subject']
            
        df = df.sort_values(by='age')
        df['y_subject'] = None
        for i, subj in enumerate(df['dsubj'].unique()):
            df.loc[df['dsubj']==subj, 'y_subject'] = i
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 12))
        
        for ax_id, x_axis in enumerate(['age', f'time_since_{disease}']):            
            sns.lineplot(
                data=df,
                x=x_axis, y='y_subject',
                units="dsubj",
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
                palette=['tab:green', 'tab:orange', 'tab:red'],
                alpha=1,
                ax=axes[ax_id]
                )
            axes[ax_id].set_xlabel(f'{x_axis} (years)', fontsize=16, fontfamily='DejaVu Sans')
            axes[ax_id].set_ylabel('Subject', fontsize=16, fontfamily='DejaVu Sans')
        fig.savefig(png, dpi=300)
    
    def get_subsets(self, df, disease='MCI', method='index cut', num_subsets=11):
        
        if 'dsubj' not in df.columns:
            df['dsubj'] = df['dataset'] + '_' + df['subject']
            
        dict_subsets = {i: pd.DataFrame() for i in range(num_subsets)}
        
        if method == 'index cut':
            for i in range(num_subsets):
                for subj in df.loc[df[f'time_since_{disease}'].notna(), 'dsubj'].unique():
                    rows_subj = df.loc[(df['dsubj']==subj) & (df[f'time_since_{disease}']<=0), ].copy()
                    rows_subj = rows_subj.sort_values(by=f'time_since_{disease}', ascending=False)
                    
                    num_dp = len(rows_subj)
                    assert num_dp >= 2, f"The number of qualified data points for {subj} is less than 2."
                    
                    dict_subsets[i] = pd.concat([dict_subsets[i], rows_subj.iloc[round(i*(num_dp-1)/(num_subsets-1)), ].to_frame().T])

        return dict_subsets