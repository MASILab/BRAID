import pdb
import pandas as pd
from pathlib import Path
from tqdm import tqdm

class DataPreparation:
    def __init__(self, dict_models, databank_csv):
        self.dict_models = dict_models
        self.databank = pd.read_csv(databank_csv)
        
    def load_data(self, folds=[1,2,3,4,5]):
        """ load dataframes from each fold of each model and combine them into a single dataframe,
        with diagnosis information collected from the databank.
        """
        
        for i, model in enumerate(self.dict_models.keys()):
            pred_root = Path(self.dict_models[model]['prediction_root'])
            col_suffix = self.dict_models[model]['col_suffix']
            
            for fold_idx in tqdm(folds, desc=f'Load data for {model}'):
                pred_csv = pred_root / f"predicted_age_fold-{fold_idx}_test_bc.csv"
                if fold_idx == 1:
                    df_model = pd.read_csv(pred_csv)
                    df_model = df_model.groupby(['dataset','subject','session','age_gt'])['age_pred'].mean().reset_index()
                    df_model = df_model.rename(columns={'age_pred': f'age_pred{col_suffix}_{fold_idx}'})
                else:
                    tmp = pd.read_csv(pred_csv)
                    tmp = tmp.groupby(['dataset','subject','session','age_gt'])['age_pred'].mean().reset_index()
                    tmp = tmp.rename(columns={'age_pred': f'age_pred{col_suffix}_{fold_idx}'})
                    df_model = df_model.merge(tmp, on=['dataset','subject','session','age_gt'])
            df_model[f'age_pred{col_suffix}_mean'] = df_model[[f'age_pred{col_suffix}_{fold_idx}' for fold_idx in folds]].mean(axis=1)
            
            if i == 0:
                df = df_model.copy()
            else:
                df = df.merge(df_model.copy(), on=['dataset','subject','session','age_gt'])
        
        df['diagnosis'] = df.apply(lambda row: self.databank.loc[
            (self.databank['dataset'] == row['dataset']) &
            (self.databank['subject'] == row['subject']) &
            ((self.databank['session'] == row['session']) | (self.databank['session'].isnull())), 'diagnosis_simple'].values[0], axis=1)
        return df

    def take_cross_sectional_samples(self, df):
        df = df.loc[(df['age_gt']>=45)&
                    (df['age_gt']<=90)&
                    (df['diagnosis'].isin(['normal', 'MCI', 'dementia'])), ].copy()
        
        df['category'] = None
        df['dsubj'] = df['dataset'] + '_' + df['subject']
        # first session of the disease-free subjects (with at least one follow-up session)
        for subj in df.loc[df['diagnosis']=='normal', 'dsubj'].unique():
            if (len(df.loc[df['dsubj']==subj, 'diagnosis'].unique()) == 1) and (len(df.loc[df['dsubj']==subj, 'age_gt'].unique()) >= 2):
                df.loc[(df['dsubj'] == subj) & (df['age_gt'] == df.loc[df['dsubj'] == subj, 'age_gt'].min()), 'category'] = 'CN'

        # session after which the subject converted to MCI or dementia
        for subj in df.loc[df['diagnosis'].isin(['MCI', 'dementia']), 'dsubj'].unique():
            rows_subj = df.loc[df['dsubj']==subj, ].copy()
            if 'normal' in rows_subj['diagnosis'].values:
                rows_subj = rows_subj.sort_values(by='age_gt')
                for i in range(len(rows_subj.index)-1):
                    if (rows_subj.iloc[i]['diagnosis'] == 'normal') & (rows_subj.iloc[i+1]['diagnosis'] in ['MCI', 'dementia']):
                        df.loc[(df['dsubj'] == subj) & 
                               (df['age_gt'] == rows_subj.iloc[i]['age_gt']), 'category'] = 'CN*'
                        break
        
        # the last session of the MCI subjects
        for subj in df.loc[df['diagnosis']=='MCI', 'dsubj'].unique():
            df.loc[(df['dsubj'] == subj) &
                   (df['age_gt'] == df.loc[(df['dsubj'] == subj)&(df['diagnosis']=='MCI'), 'age_gt'].max()), 'category'] = 'MCI'
        
        # the last session of the dementia subjects
        for subj in df.loc[df['diagnosis']=='dementia', 'dsubj'].unique():
            df.loc[(df['dsubj'] == subj) &
                   (df['age_gt'] == df.loc[(df['dsubj'] == subj)&(df['diagnosis']=='dementia'), 'age_gt'].max()), 'category'] = 'AD'
        
        df = df.dropna(subset=['category'])
        return df
        
    def over_sampling(self, df, column_class='category', random_state=0):
        """ over-sampling the minority classes to balance the dataset """
        num_max = df[column_class].value_counts().max()
        for c in df[column_class].unique():
            num_ori = df[column_class].value_counts()[c]
            if num_ori < num_max:
                new_samples = df.loc[df[column_class] == c, ].sample(n=num_max-num_ori, replace=True, random_state=random_state)
                df = pd.concat([df, new_samples], axis=0)
        return df

    def subject_level_splitting(self, df, num_folds=5, column_name='fold_idx', random_state=0):
        """ split the dataset at the subject level for cross-validation,
        save the fold information in a seperate column
        """
        assert column_name not in df.columns, f'Column {column_name} already exists in the dataframe'

        df[column_name] = 1  # we will sample num_folds-1 times, and the rest will be fold-1

        for c in df['category'].unique():
            num_total = len(df.loc[df['category']==c, ].index)
            num_per_fold = num_total // num_folds
            for fold_idx in range(2, num_folds+1):
                dsubj_fold = df['dsubj'].sample(n=num_per_fold, replace=False, random_state=random_state).values
                df.loc[df['dsubj'].isin(dsubj_fold), column_name] = fold_idx
        return df
        
dict_models = {
    'WM age model': {
        'prediction_root': 'models/2024-02-07_ResNet101_BRAID_warp/predictions',
        'col_suffix': '_wm_age',
        },
    'GM age model (ours)': {
        'prediction_root': 'models/2024-02-07_T1wAge_ResNet101/predictions',
        'col_suffix': '_gm_age_ours'
        },
    'WM age model (contaminated with GM age features)': {
        'prediction_root': 'models/2023-12-22_ResNet101/predictions',
        'col_suffix': '_wm_age_contaminated'
        },
    'GM age model (TSAN)': {
        'prediction_root': 'models/2024-02-12_TSAN_first_stage/predictions',
        'col_suffix': '_gm_age_tsan'
        },
}

d = DataPreparation(dict_models, databank_csv='/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv')
df = d.load_data()
df = d.take_cross_sectional_samples(df)
df = d.over_sampling(df)
df = d.subject_level_splitting(df)

# TODO: data preparation is done, please move forward with feature/model configuration