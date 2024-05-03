import pdb
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from warnings import simplefilter
from sklearn.utils import resample
from scipy.stats import chi2
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def roster_brain_age_models():
    dict_models = {
        'WM age nonlinear': {
            'prediction_root': 'models/2024-02-07_ResNet101_BRAID_warp/predictions',
            'col_suffix': '_wm_age_nonlinear',
        },
        'GM age (ours)': {
            'prediction_root': 'models/2024-02-07_T1wAge_ResNet101/predictions',
            'col_suffix': '_gm_age_ours',
        },
        'WM age affine': {
            'prediction_root': 'models/2023-12-22_ResNet101/predictions',
            'col_suffix': '_wm_age_affine',
        },
        'GM age (TSAN)': {
            'prediction_root': 'models/2024-02-12_TSAN_first_stage/predictions',
            'col_suffix': '_gm_age_tsan',
        },
        'GM age (DeepBrainNet)': {
            'prediction_root': 'models/2024-04-04_DeepBrainNet/predictions',
            'col_suffix': '_gm_age_dbn',
        },
    }
    return dict_models


class CoxPHDataPrep():
    def __init__(self, dict_models, databank_csv, bias_correction=True):
        self.dict_models = dict_models
        self.databank = pd.read_csv(databank_csv)
        self.bias_correction = bias_correction

    def load_predictions_of_all_models(self):
        bc = '_bc' if self.bias_correction else ''
        
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
            print(f'Loaded data for {model}, shape: {df_model.shape}')
            
            if i == 0:
                df = df_model.copy()
            else:
                df = df.merge(df_model.copy(), on=['dataset','subject','session','sex','age'])
        
        # remove duplicated rows (e.g., ses-270scanner09, ses-270scanner10 in BLSA)
        df = df.sort_values(by=['dataset', 'subject', 'age', 'session'], ignore_index=True)
        df = df.drop_duplicates(subset=['dataset', 'subject', 'age'], keep='last', ignore_index=True)
        print(f"--------> Predictions loaded. DataFrame shape: {df.shape}")
        return df

    def retrieve_diagnosis_label(self, df):
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

    def feature_engineering(self, df):
        """ Create new features from current data. Convert categorical data to binary.
        """
        # Convert sex to binary
        df['sex'] = df['sex'].map({'female': 0, 'male': 1})

        # Mean value of age predictions from all folds for each model
        for model in self.dict_models.keys():
            if model=='GM age (DeepBrainNet)': 
                continue  # because there is only one model
            else:
                col_suffix = self.dict_models[model]['col_suffix']
                df[f'age_pred{col_suffix}_mean'] = df[[f'age_pred{col_suffix}_{fold_idx}' for fold_idx in [1,2,3,4,5]]].mean(axis=1)
        
        # Brain age gap (BAG)
        for model in self.dict_models.keys():
            col_suffix = self.dict_models[model]['col_suffix']
            if model=='GM age (DeepBrainNet)':
                df[f'age_pred{col_suffix}_bag'] = df[f'age_pred{col_suffix}'] - df['age']
            else:
                for fold_idx in [1,2,3,4,5]:
                    df[f'age_pred{col_suffix}_{fold_idx}_bag'] = df[f'age_pred{col_suffix}_{fold_idx}'] - df['age']
                df[f'age_pred{col_suffix}_mean_bag'] = df[f'age_pred{col_suffix}_mean'] - df['age']
        
        # BAG change rate_i = (BAG_i+1 - BAG_i) / (age_i+1 - age_i)
        for model in self.dict_models.keys():
            col_suffix = self.dict_models[model]['col_suffix']
            if model=='GM age (DeepBrainNet)':
                df[f'age_pred{col_suffix}_bag_change_rate'] = None
            else:
                for fold_idx in [1,2,3,4,5]:
                    df[f'age_pred{col_suffix}_{fold_idx}_bag_change_rate'] = None
                df[f'age_pred{col_suffix}_mean_bag_change_rate'] = None
        
        for subj in df['subj'].unique():
            rows_subj = df.loc[df['subj']==subj, ].copy()
            rows_subj = rows_subj.sort_values(by='age')
            for i in range(len(rows_subj.index)-1):
                interval = rows_subj.iloc[i+1]['age'] - rows_subj.iloc[i]['age']  # age_i+1 - age_i
                
                for model in self.dict_models.keys():
                    col_suffix = self.dict_models[model]['col_suffix']
                    if model=='GM age (DeepBrainNet)':
                        delta_bag = rows_subj.iloc[i+1][f'age_pred{col_suffix}_bag'] - rows_subj.iloc[i][f'age_pred{col_suffix}_bag']
                        df.loc[(df['subj']==subj)&(df['age']==rows_subj.iloc[i]['age']), f'age_pred{col_suffix}_bag_change_rate'] = (delta_bag / interval) if interval > 0 else None
                    else:
                        for fold_idx in [1,2,3,4,5]:
                            delta_bag = rows_subj.iloc[i+1][f'age_pred{col_suffix}_{fold_idx}_bag'] - rows_subj.iloc[i][f'age_pred{col_suffix}_{fold_idx}_bag']
                            df.loc[(df['subj']==subj)&(df['age']==rows_subj.iloc[i]['age']), f'age_pred{col_suffix}_{fold_idx}_bag_change_rate'] = (delta_bag / interval) if interval > 0 else None
                        delta_bag = rows_subj.iloc[i+1][f'age_pred{col_suffix}_mean_bag'] - rows_subj.iloc[i][f'age_pred{col_suffix}_mean_bag']
                        df.loc[(df['subj']==subj)&(df['age']==rows_subj.iloc[i]['age']), f'age_pred{col_suffix}_mean_bag_change_rate'] = (delta_bag / interval) if interval > 0 else None
        
        # interactions (chronological age/sex with BAG/BAG change rate)
        for model in self.dict_models.keys():
            col_suffix = self.dict_models[model]['col_suffix']
            if model=='GM age (DeepBrainNet)':
                df[f'age_pred{col_suffix}_bag_multiply_age'] = df[f'age_pred{col_suffix}_bag'] * df['age']
                df[f'age_pred{col_suffix}_bag_change_rate_multiply_age'] = df[f'age_pred{col_suffix}_bag_change_rate'] * df['age']
                df[f'age_pred{col_suffix}_bag_multiply_sex'] = df[f'age_pred{col_suffix}_bag'] * df['sex']
                df[f'age_pred{col_suffix}_bag_change_rate_multiply_sex'] = df[f'age_pred{col_suffix}_bag_change_rate'] * df['sex']
            else:
                for fold_idx in [1,2,3,4,5]:
                    df[f'age_pred{col_suffix}_{fold_idx}_bag_multiply_age'] = df[f'age_pred{col_suffix}_{fold_idx}_bag'] * df['age']
                    df[f'age_pred{col_suffix}_{fold_idx}_bag_change_rate_multiply_age'] = df[f'age_pred{col_suffix}_{fold_idx}_bag_change_rate'] * df['age']
                    df[f'age_pred{col_suffix}_{fold_idx}_bag_multiply_sex'] = df[f'age_pred{col_suffix}_{fold_idx}_bag'] * df['sex']
                    df[f'age_pred{col_suffix}_{fold_idx}_bag_change_rate_multiply_sex'] = df[f'age_pred{col_suffix}_{fold_idx}_bag_change_rate'] * df['sex']
                df[f'age_pred{col_suffix}_mean_bag_multiply_age'] = df[f'age_pred{col_suffix}_mean_bag'] * df['age']
                df[f'age_pred{col_suffix}_mean_bag_change_rate_multiply_age'] = df[f'age_pred{col_suffix}_mean_bag_change_rate'] * df['age']
                df[f'age_pred{col_suffix}_mean_bag_multiply_sex'] = df[f'age_pred{col_suffix}_mean_bag'] * df['sex']
                df[f'age_pred{col_suffix}_mean_bag_change_rate_multiply_sex'] = df[f'age_pred{col_suffix}_mean_bag_change_rate'] * df['sex']

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

    def sample_subjects_for_coxph(self, df, disease):
        # Baseline session of subject who converted from CN to MCI/AD
        df_event = pd.DataFrame()
        for subj in df.loc[df[f'time_to_{disease}']>=0, 'subj'].unique():
            rows_subj = df.loc[df['subj']==subj, ].copy()
            assert len(rows_subj.index) > 1, f"Subject {subj} ({disease}) has only one session."    
            df_event = pd.concat([df_event, rows_subj.loc[rows_subj['age']==rows_subj['age'].min(), ]], ignore_index=True)
        df_event['time_to_event'] = df_event[f'time_to_{disease}']
        df_event['event'] = 1
        
        # Baseline session of subject who stayed cognitively normal until the last session
        df_censored = pd.DataFrame()
        for subj in df.loc[df['dataset'].isin(df_event['dataset'].unique()) & (df['cn_label']==1), 'subj'].unique():
            rows_subj = df.loc[df['subj']==subj, ].copy()
            age_baseline = rows_subj['age'].min()
            if age_baseline > df_event['age'].max():
                continue
            if age_baseline < (df_event['age'].min()-1):
                continue
            df_censored = pd.concat([df_censored, rows_subj.loc[rows_subj['age']==age_baseline, ]], ignore_index=True)
        df_censored['time_to_event'] = df_censored['time_to_last_cn']
        df_censored['event'] = 0

        # Combine event and censored data
        df_cox = pd.concat([df_event, df_censored], ignore_index=True)

        # Print out the life table
        headers = ['Interval (years)', 'Number CN at Beginning of Interval', f'Number of {disease} During Interval', 'Number Censored']
        table_data = []
        window_start = 0
        window_size = 2
        while window_start < df_cox['time_to_event'].max():
            window_end = window_start + window_size
            num_cn = len(df_cox.loc[df_cox['time_to_event']>=window_start, ].index)
            num_event = len(df_cox.loc[(df_cox['time_to_event']>=window_start) & (df_cox['time_to_event']<window_end) & (df_cox['event']==1), ].index)
            num_censored = len(df_cox.loc[(df_cox['time_to_event']>=window_start) & (df_cox['time_to_event']<window_end) & (df_cox['event']==0), ].index)
            table_data.append([f'{window_start}-{window_end}', num_cn, num_event, num_censored])
            window_start = window_end
        print(tabulate(table_data, headers, tablefmt='grid'))
        
        return df_cox
    
    def get_dataframe(self, disease='MCI'):
        df = self.load_predictions_of_all_models()
        df = self.retrieve_diagnosis_label(df)
        df = self.assign_cn_label(df)
        df = self.feature_engineering(df)
        df = self.mark_progression_subjects_out(df)
        df_cox = self.sample_subjects_for_coxph(df, disease)
        return df_cox


def roster_features(df):
    feat_combo = {'basic: chronological age + sex': ['age', 'sex']}
    # feat_combo['basic + GM age (ours)'] = ['age', 'sex'] + [col for col in df.columns if ('_gm_age_ours' in col) and ('_mean_bag' in col)]
    # feat_combo['basic + GM age (TSAN)'] = ['age', 'sex'] + [col for col in df.columns if ('_gm_age_tsan' in col) and ('_mean_bag' in col)]
    # feat_combo['basic + GM age (DeepBrainNet)'] = ['age', 'sex'] + [col for col in df.columns if ('_gm_age_dbn' in col) and ('_bag' in col)]
    # feat_combo['WM age nonlinear'] = [col for col in df.columns if ('_wm_age_nonlinear' in col) and ('_mean_bag' in col)]
    feat_combo['basic + GM age (ours)'] = ['age', 'sex'] + ['age_pred_gm_age_ours_mean_bag']
    feat_combo['basic + GM age (TSAN)'] = ['age', 'sex'] + ['age_pred_gm_age_tsan_mean_bag']
    feat_combo['basic + GM age (DeepBrainNet)'] = ['age', 'sex'] + ['age_pred_gm_age_dbn_bag']
    feat_combo['WM age nonlinear'] = ['age_pred_wm_age_nonlinear_mean_bag']

    for feat_combo_name, feat_cols in feat_combo.items():
        print(f'{feat_combo_name}:\n{feat_cols}\n')
    return feat_combo


def bootstrap_c_index(df_cox, predicted_hazard, num_bootstrap=1000):
    idx = np.arange(len(df_cox))
    c_indexes = []

    for _ in range(num_bootstrap):
        resample_idx = resample(idx)
        y_true = df_cox.loc[resample_idx, 'time_to_event']
        y_event = df_cox.loc[resample_idx, 'event']
        y_pred = predicted_hazard.loc[resample_idx]

        c_index = concordance_index(y_true, -y_pred, y_event)
        c_indexes.append(c_index)

    c_index_ci_lower = np.percentile(c_indexes, 2.5)
    c_index_ci_upper = np.percentile(c_indexes, 97.5)
    c_index_mean = np.mean(c_indexes)
    return c_index_mean, c_index_ci_lower, c_index_ci_upper


if __name__ == '__main__':
    # Data preparation for Cox PH model
    output_fn = 'experiments/2024-05-02_Cox_PH_Models/data/data_cox_ph.csv'
    if Path(output_fn).is_file():
        df_cox = pd.read_csv(output_fn)
    else:
        cox_data_prep = CoxPHDataPrep(roster_brain_age_models(), '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv')
        df_cox = cox_data_prep.get_dataframe(disease='MCI')
        df_cox.to_csv(output_fn, index=False)
    feat_combo = roster_features(df_cox)

    # Cox PH model with different feature combinations
    headers = ['features', 'c-index (wo/ WM age)', 'c-index (w/ WM age)', 'AIC (wo/ WM age)', 'AIC (w/ WM age)', 'chi2', 'p-value']
    table_data = []

    base_feat = [
        'basic: chronological age + sex',
        'basic + GM age (ours)',
        'basic + GM age (TSAN)',
        'basic + GM age (DeepBrainNet)',
    ]
    for base_feat_name in base_feat:
        base_model = CoxPHFitter()
        base_model.fit(df_cox[feat_combo[base_feat_name] + ['time_to_event', 'event']], duration_col='time_to_event', event_col='event')
        base_pred = base_model.predict_partial_hazard(X=df_cox[feat_combo[base_feat_name]])
        base_c_index_mean, base_c_index_ci_lower, base_c_index_ci_upper = bootstrap_c_index(df_cox, base_pred)
        base_dof = base_model.params_.shape[0]  #degree of freedom
        base_aic = base_model.AIC_partial_   # partial AIC

        full_model = CoxPHFitter()
        full_model.fit(df_cox[feat_combo[base_feat_name] + feat_combo['WM age nonlinear'] + ['time_to_event', 'event']], duration_col='time_to_event', event_col='event')
        full_pred = full_model.predict_partial_hazard(X=df_cox[feat_combo[base_feat_name] + feat_combo['WM age nonlinear']])
        full_c_index_mean, full_c_index_ci_lower, full_c_index_ci_upper = bootstrap_c_index(df_cox, full_pred)
        full_dof = full_model.params_.shape[0]
        full_aic = full_model.AIC_partial_
        
        test_statistic = 2 * (full_model.log_likelihood_ - base_model.log_likelihood_)
        df = full_dof - base_dof
        p_value = chi2.sf(test_statistic, df)
        
        table_data.append([
            base_feat_name, 
            f'{base_c_index_mean:.2f} ({base_c_index_ci_lower:.2f}, {base_c_index_ci_upper:.2f})', 
            f'{full_c_index_mean:.2f} ({full_c_index_ci_lower:.2f}, {full_c_index_ci_upper:.2f})', 
            f'{base_aic:.1f}', f'{full_aic:.1f}', f'{test_statistic:.2f}', p_value])

    print(tabulate(table_data, headers, tablefmt='grid'))