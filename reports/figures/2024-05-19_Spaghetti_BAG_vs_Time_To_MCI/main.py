import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

def roster_brain_age_models():
    dict_models = {
        'WM age nonlinear': {
            'prediction_root': 'models/2024-02-07_ResNet101_BRAID_warp/predictions',
            'col_suffix': '_wm_age_nonlinear',
            'style': {'color': (0,0,1), 'alpha': 1, 'linestyle': 'solid'},
        },
        'WM age affine': {
            'prediction_root': 'models/2023-12-22_ResNet101/predictions',
            'col_suffix': '_wm_age_affine',
            'style': {'color': (0.5,0,1), 'alpha': 1, 'linestyle': 'solid'},
        },
        'GM age (ours)': {
            'prediction_root': 'models/2024-02-07_T1wAge_ResNet101/predictions',
            'col_suffix': '_gm_age_ours',
            'style': {'color': (1,0,0), 'alpha': 1, 'linestyle': 'solid'},
        },
        'GM age (DeepBrainNet)': {
            'prediction_root': 'models/2024-04-04_DeepBrainNet/predictions',
            'col_suffix': '_gm_age_dbn',
            'style': {'color': (1,0.5,0.5), 'alpha': 1, 'linestyle': 'dashed'},
        },
        'GM age (TSAN)': {
            'prediction_root': 'models/2024-02-12_TSAN_first_stage/predictions',
            'col_suffix': '_gm_age_tsan',
            'style': {'color': (0.5,0,0), 'alpha': 1, 'linestyle': 'dotted'},
        },
    }
    return dict_models


class DataPreparation:
    def __init__(self, dict_models, databank_csv):
        self.dict_models = dict_models
        self.databank = pd.read_csv(databank_csv)
    
    def load_predictions_of_all_models(self, bias_correction=True):
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
                for fold_idx in [1,2,3,4,5]:
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
                df_model[f'age_pred{col_suffix}'] = df_model[[f'age_pred{col_suffix}_{fold_idx}' for fold_idx in [1,2,3,4,5]]].mean(axis=1)
            df_model[f'age_pred{col_suffix}_bag'] = df_model[f'age_pred{col_suffix}'] - df_model['age']
            df_model = df_model[['dataset','subject','session','sex','age',f'age_pred{col_suffix}',f'age_pred{col_suffix}_bag']]
            print(f'Loaded data for {model}, shape: {df_model.shape}')
            
            if i == 0:
                df = df_model.copy()
            else:
                df = df.merge(df_model.copy(), on=['dataset','subject','session','sex','age'])
        
        # remove duplicated rows
        df = df.sort_values(by=['dataset', 'subject', 'age', 'session'], ignore_index=True)
        df = df.drop_duplicates(subset=['dataset', 'subject', 'age'], keep='last', ignore_index=True)
        print(f"--------> Predictions loaded. DataFrame shape: {df.shape}")
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
        print(f"--------> Diagnosis labels retrieved. {len(df.loc[df['diagnosis'].isna(),].index)} out of {len(df.index)} do not have diagnosis info.")
        return df
    
    def assign_cn_label(self, df):
        """ Create the following new columns:
        "cn_label": 0.5 for cognitively normal, and has only cognitively normal in his/her diagnosis history.
            1 for all above, plus has at least one following session in which the subject is still cognitively normal.
        "age_last_cn": the age of the last available session of the subject (cn_label >= 0.5).
        "time_to_last_cn": the time (in years) to the "age_last_cn".
        """
        if 'subj' not in df.columns:
            df['subj'] = df['dataset'] + '_' + df['subject']
        
        df['cn_label'] = None
        df['age_last_cn'] = None
        df['time_to_last_cn'] = None

        for subj in df.loc[df['diagnosis']=='normal', 'subj'].unique():
            if len(df.loc[df['subj']==subj, 'diagnosis'].unique())==1:  # there is only 'normal' in diagnosis history
                df.loc[df['subj']==subj, 'cn_label'] = 0.5
                df.loc[df['subj']==subj, 'age_last_cn'] = df.loc[df['subj']==subj, 'age'].max()
                if len(df.loc[df['subj']==subj, 'age'].unique())>=2:  # at least two sessions are available
                    # pick all but the last session (which is uncertain if it progresses to MCI/AD)
                    df.loc[(df['subj']==subj) & (df['age']!=df.loc[df['subj']==subj,'age'].max()), 'cn_label'] = 1
        df['time_to_last_cn'] = df['age_last_cn'] - df['age']

        num_subj_strict = len(df.loc[df['cn_label']==1, 'subj'].unique())
        num_subj_loose = len(df.loc[df['cn_label']>=0.5, 'subj'].unique())
        print(f'--------> Found {num_subj_strict} subjects with strict CN label, and {num_subj_loose} subjects with loose CN label.')
        return df
    
    def mark_progression_subjects_out(self, df):
        """ Create the following columns to the dataframe:
            - "age_AD": the age when the subject was diagnosed with AD for the first time.
            - "time_to_AD": the time (in years) to the first AD diagnosis.
            - "age_MCI": the age when the subject was diagnosed with MCI for the first time.
            - "time_to_MCI": the time (in years) to the first MCI diagnosis.
        Subjects with following characteristics are excluded:
            - the diagnosis of available sessions starts with MCI or AD.
            - the subject turned back to cognitively normal after being diagnosed with MCI or AD.
        """
        df = df.loc[df['diagnosis'].isin(['normal', 'MCI', 'AD']), ].copy()

        if 'subj' not in df.columns:
            df['subj'] = df['dataset'] + '_' + df['subject']
        
        for disease in ['AD', 'MCI']:
            df[f'age_{disease}'] = None
            
            for subj in df.loc[df['diagnosis']==disease, 'subj'].unique():
                include_this_subj = True
                rows_subj = df.loc[df['subj']==subj, ].copy()
                rows_subj = rows_subj.sort_values(by='age')
                if rows_subj.iloc[0]['diagnosis'] != 'normal':
                    include_this_subj = False
                for i in range(len(rows_subj.index)-1):
                    if rows_subj.iloc[i]['diagnosis']==disease and rows_subj.iloc[i+1]['diagnosis']=='normal':
                        include_this_subj = False
                        break
                if include_this_subj:
                    df.loc[df['subj']==subj, f'age_{disease}'] = rows_subj.loc[rows_subj['diagnosis']==disease, 'age'].min()
            df[f'time_to_{disease}'] = df[f'age_{disease}'] - df['age']
            
            num_subj = len(df.loc[df[f'age_{disease}'].notna(), 'subj'].unique())
            print(f'--------> Found {num_subj} subjects with {disease} progression.')
        
        return df

if __name__ == '__main__':
    # Data Preparation
    data_prep = DataPreparation(roster_brain_age_models(), '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv')
    output_fn = 'reports/figures/2024-05-19_Spaghetti_BAG_vs_Time_To_MCI/data/data.csv'
    if Path(output_fn).is_file():
        df = pd.read_csv(output_fn)
    else:
        df = data_prep.load_predictions_of_all_models(bias_correction=True)
        df = data_prep.retrieve_diagnosis_label(df)
        df = data_prep.assign_cn_label(df)
        df = data_prep.mark_progression_subjects_out(df)
        df = df.loc[df['time_to_MCI'].notna(), ].copy()
        df.to_csv(output_fn, index=False)

    # Linear mixed-effects models: BAG ~ time_to_mci + time_to_mci^2 + (1|subj)
    df['time_to_MCI_square'] = df['time_to_MCI'] ** 2
    for model in data_prep.dict_models.keys():
        col_suffix = data_prep.dict_models[model]['col_suffix']
        col_bag = f'age_pred{col_suffix}_bag'
        
        lme = smf.mixedlm(f"{col_bag} ~ time_to_MCI + time_to_MCI_square", data=df, groups=df["subj"])
        result = lme.fit()
        x = sorted(df['time_to_MCI'].unique())
        y = [result.fe_params['Intercept'] + result.fe_params['time_to_MCI']*i + result.fe_params['time_to_MCI_square']*i**2 for i in x]
        
        fig, ax = plt.subplots(1,1,figsize=(6.5, 5))
        sns.lineplot(
            data=df, x='time_to_MCI', y=col_bag, units='subj', estimator=None, 
            lw=0.5, color='tab:gray', alpha=0.5, linestyle='-', ax=ax
            )
        sns.lineplot(x=x, y=y, color=data_prep.dict_models[model]['style']['color'], lw=2, ax=ax)
        
        ax.set_ylim(-20, 20)
        ax.invert_xaxis()
        fig.savefig(f'reports/figures/2024-05-19_Spaghetti_BAG_vs_Time_To_MCI/figs/{col_bag}.png', dpi=300)
        plt.close(fig)