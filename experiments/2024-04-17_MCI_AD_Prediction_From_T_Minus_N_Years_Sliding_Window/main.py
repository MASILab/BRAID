import pdb
import textwrap
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from tqdm import tqdm
from pathlib import Path
from warnings import simplefilter
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
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


def roster_feature_combinations(df):
    feat_combo = {'basic: chronological age + sex': ['age', 'sex']}
    feat_combo['basic + WM age nonlinear'] = ['age', 'sex'] + [col for col in df.columns if '_wm_age_nonlinear' in col]
    feat_combo['basic + GM age (ours)'] = ['age', 'sex'] + [col for col in df.columns if '_gm_age_ours' in col]
    feat_combo['basic + GM age (TSAN)'] = ['age', 'sex'] + [col for col in df.columns if '_gm_age_tsan' in col]
    feat_combo['basic + GM age (DeepBrainNet)'] = ['age', 'sex'] + [col for col in df.columns if '_gm_age_dbn' in col]
    feat_combo['basic + WM age affine'] = ['age', 'sex'] + [col for col in df.columns if '_wm_age_affine' in col]
    feat_combo['basic + WM age nonlinear + GM age (ours)'] = ['age', 'sex'] + [col for col in df.columns if '_wm_age_nonlinear' in col] + [col for col in df.columns if '_gm_age_ours' in col]
    feat_combo['basic + WM age nonlinear + GM age (TSAN)'] = ['age', 'sex'] + [col for col in df.columns if '_wm_age_nonlinear' in col] + [col for col in df.columns if '_gm_age_tsan' in col]
    feat_combo['basic + WM age nonlinear + GM age (DeepBrainNet)'] = ['age', 'sex'] + [col for col in df.columns if '_wm_age_nonlinear' in col] + [col for col in df.columns if '_gm_age_dbn' in col]

    for feat_combo_name, feat_cols in feat_combo.items():
        print(f'{feat_combo_name}:\n{feat_cols}\n')
    return feat_combo


def roster_classifiers():
    classifiers = {
        'Logistic Regression': (LogisticRegression, {'random_state': 42, 'max_iter': 1000}),
        'Linear SVM': (SVC, {'kernel': "linear", 'C': 1, 'probability': True, 'random_state': 42}),
        'Random Forest': (RandomForestClassifier, {'n_estimators': 100, 'random_state': 42}),
    }
    return classifiers


class DataPreparation:
    def __init__(self, dict_models, databank_csv):
        self.dict_models = dict_models
        self.databank = pd.read_csv(databank_csv)
    
    def load_predictions_of_all_models(self, bias_correction=True):
        """ Load dataframes from each fold (if cross-validation was used) of each model 
        and combine them into a single wide dataframe.
        If bias_correction is False, load the results without bias correction.
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
        
    def get_matched_cn_data(self, df_master, df_subset, disease, time_diff_threshold=1, mode='hungry_but_picky', match_dataset=False):
        """ Use greedy algorithm to sample a subset of cognitively normal (CN) data points from the main dataframe df_master.
        The subset matches the data points in df_subset in terms of age, sex, dataset (when match_dataset=True), and 
        time_to_{event, i.e., first MCI/AD or last CN} (when mode="hungry_but_picky").
            
            The following modes (matching strategies) are available (ordered by the time it was implemented):
            - "lavish": finds matches based on the age, sex, and dataset (when match_dataset=True). It prioritizes using 
        the subjects that are CN under the strict definition (cn_label==1). If it does not find any match, it will then use 
        the subjects that are CN under the loose definition (cn_label==0.5). Data points with the closest age will be used. 
        Only one data point from the same subject is allowed to be used. And that is why it is called "lavish".
            - "allow_multiple_per_subject": follows the same rules as the "lavish" mode, except that it is allowed to use
        more than one data points from the same subject as long as these data points are used to match the data points of 
        the same subject in df_subset.
            - "hungry": finds matches based on the age, sex, and dataset (when match_dataset=True). It tries to match as many 
        data points as possible. It will start from the subjects in the df_subset that have the most data points to be matched, 
        and use the subject in df_master that has the most matched data points to match with.
            - "hungry_but_picky": follows the same rules as the "hungry" mode, except that it will add one more criteria for
        matching: the time_to_last_cn should match with the time_to_{disease}.
        
            The matched subset will be concatenated with df_subset and returned. There will be two new columns:
            - "clf_label": 1 for disease, and 0 for cognitively normal.
            - "match_id": the ID of the matched pair. (could be useful for pair-level split)
        """
        if mode in ['hungry_but_picky', 'hungry']:
            df_candidates = df_master.loc[df_master['cn_label']>=0.5, ].copy()
            df_candidates['clf_label'] = 0
            df_candidates['match_id'] = None
            df_subset['clf_label'] = 1
            df_subset['match_id'] = None
            num_dp_total = len(df_subset.index)

            match_id = 0
            subj = None
            subjs_done = []

            while True:
                todo_subjects = df_subset.loc[df_subset['match_id'].isna(), ].groupby('subj').size().sort_values(ascending=False).index
                if len(todo_subjects) == 0:
                    print('Data matching done.')
                    break
                else:
                    todo_subjects = [s for s in todo_subjects if s not in subjs_done]
                    if len(todo_subjects) == 0:
                        print('Data matching done.')
                        break
                    else:
                        subj = todo_subjects[0]
                
                # find the candidate subject that has the most matched data points for the current subj                
                subj_c_most_match = None
                subj_c_most_match_num = 0
                for subj_c in df_candidates['subj'].unique():
                    if subj_c in df_candidates.loc[df_candidates['match_id'].notna(), 'subj'].unique():
                        continue  # because already used in previous match
                    if df_candidates.loc[df_candidates['subj']==subj_c, 'sex'].values[0] != df_subset.loc[df_subset['subj']==subj, 'sex'].values[0]:
                        continue
                    if (df_candidates.loc[df_candidates['subj']==subj_c, 'dataset'].values[0] != df_subset.loc[df_subset['subj']==subj, 'dataset'].values[0]) and match_dataset:
                        continue
                    
                    matched_age_c = []
                    if mode == 'hungry':
                        for age in sorted(df_subset.loc[(df_subset['subj']==subj)&df_subset['match_id'].isna(), 'age'].unique()):
                            for age_c in sorted(df_candidates.loc[(df_candidates['subj']==subj_c)&df_candidates['match_id'].isna(), 'age'].unique()):
                                if (abs(age_c - age) < time_diff_threshold) and (age_c not in matched_age_c):
                                    matched_age_c.append(age_c)
                                    break
                    else:
                        for age in sorted(df_subset.loc[(df_subset['subj']==subj)&df_subset['match_id'].isna(), 'age'].unique()):
                            time_to_disease = df_subset.loc[(df_subset['subj']==subj)&(df_subset['age']==age), f'time_to_{disease}'].values[0]
                            for age_c in sorted(df_candidates.loc[(df_candidates['subj']==subj_c)&df_candidates['match_id'].isna(), 'age'].unique()):
                                time_to_last_cn = df_candidates.loc[(df_candidates['subj']==subj_c)&(df_candidates['age']==age_c), 'time_to_last_cn'].values[0]
                                if (abs(age_c - age) < time_diff_threshold) and (abs(time_to_last_cn - time_to_disease) < time_diff_threshold) and (age_c not in matched_age_c):
                                    matched_age_c.append(age_c)
                                    break

                    if len(matched_age_c) > subj_c_most_match_num:
                        subj_c_most_match = subj_c
                        subj_c_most_match_num = len(matched_age_c)
                        
                if subj_c_most_match is not None:
                    ct_add = 0
                    matched_age_c = []
                    if mode == 'hungry':
                        for age in sorted(df_subset.loc[(df_subset['subj']==subj)&df_subset['match_id'].isna(), 'age'].unique()):
                            for age_c in sorted(df_candidates.loc[(df_candidates['subj']==subj_c_most_match)&df_candidates['match_id'].isna(), 'age'].unique()):
                                if (abs(age_c - age) < time_diff_threshold) and (age_c not in matched_age_c):
                                    matched_age_c.append(age_c)
                                    df_subset.loc[(df_subset['subj']==subj)&(df_subset['age']==age), 'match_id'] = match_id
                                    df_candidates.loc[(df_candidates['subj']==subj_c_most_match)&(df_candidates['age']==age_c), 'match_id'] = match_id
                                    match_id += 1
                                    ct_add += 1
                                    break
                    else:
                        for age in sorted(df_subset.loc[(df_subset['subj']==subj)&df_subset['match_id'].isna(), 'age'].unique()):
                            time_to_disease = df_subset.loc[(df_subset['subj']==subj)&(df_subset['age']==age), f'time_to_{disease}'].values[0]
                            for age_c in sorted(df_candidates.loc[(df_candidates['subj']==subj_c_most_match)&df_candidates['match_id'].isna(), 'age'].unique()):
                                time_to_last_cn = df_candidates.loc[(df_candidates['subj']==subj_c_most_match)&(df_candidates['age']==age_c), 'time_to_last_cn'].values[0]
                                if (abs(age_c - age) < time_diff_threshold) and (abs(time_to_last_cn - time_to_disease) < time_diff_threshold) and (age_c not in matched_age_c):
                                    matched_age_c.append(age_c)
                                    df_subset.loc[(df_subset['subj']==subj)&(df_subset['age']==age), 'match_id'] = match_id
                                    df_candidates.loc[(df_candidates['subj']==subj_c_most_match)&(df_candidates['age']==age_c), 'match_id'] = match_id
                                    match_id += 1
                                    ct_add += 1
                                    break

                    assert ct_add == subj_c_most_match_num, f'ct_add: {ct_add}, subj_c_most_match_num: {subj_c_most_match_num}'
                    num_dp_matched = len(df_subset.loc[df_subset['match_id'].notna(), ].index)
                    print(f"Matched {subj_c_most_match_num} data points of {subj_c_most_match}. {num_dp_matched} (matched) / {num_dp_total} (total)")
                else:
                    subjs_done.append(subj)

            df_subset_matched = pd.concat([
                df_subset.loc[df_subset['match_id'].notna(), ], 
                df_candidates.loc[df_candidates['match_id'].notna(), ]])
        
        elif mode == 'allow_multiple_per_subject':
            df_subset_matched = pd.DataFrame()
            match_id = 0
            for _, row in df_subset.iterrows():
                used_subj = [] if len(df_subset_matched.index)==0 else df_subset_matched['subj'].unique().tolist()
                
                best_match = None
                best_diff = time_diff_threshold
                
                for _, row_c in df_master.loc[df_master['cn_label']==1, ].iterrows():
                    if (row_c['sex'] != row['sex']):
                        continue
                    if (row_c['dataset'] != row['dataset']) and match_dataset:
                        continue
                    if row_c['subj'] in used_subj:
                        where_this_subj_was_used = df_subset_matched.loc[df_subset_matched['subj']==row_c['subj'], 'match_id'].unique()
                        where_this_subj_was_used = df_subset_matched.loc[df_subset_matched['match_id'].isin(where_this_subj_was_used), 'subj'].unique()
                        if set(where_this_subj_was_used) != set([row_c['subj'], row['subj']]):
                            continue
                    age_diff = abs(row_c['age'] - row['age'])
                    if age_diff < best_diff:
                        best_match = row_c
                        best_diff = age_diff
                
                if best_match is None:
                    for _, row_c in df_master.loc[df_master['cn_label']==0.5, ].iterrows():
                        if (row_c['sex'] != row['sex']):
                            continue
                        if (row_c['dataset'] != row['dataset']) and match_dataset:
                            continue
                        if row_c['subj'] in used_subj:
                            where_this_subj_was_used = df_subset_matched.loc[df_subset_matched['subj']==row_c['subj'], 'match_id'].unique()
                            where_this_subj_was_used = df_subset_matched.loc[df_subset_matched['match_id'].isin(where_this_subj_was_used), 'subj'].unique()
                            if set(where_this_subj_was_used) != set([row_c['subj'], row['subj']]):
                                continue
                        age_diff = abs(row_c['age'] - row['age'])
                        if age_diff < best_diff:
                            best_match = row_c
                            best_diff = age_diff
                
                if best_match is not None:
                    for r, clf_label in [(row, 1), (best_match, 0)]:
                        r = r.to_frame().T
                        r['clf_label'] = clf_label
                        r['match_id'] = match_id
                        df_subset_matched = pd.concat([df_subset_matched, r])
                    match_id += 1
                    
        elif mode == 'lavish':
            df_subset_matched = pd.DataFrame()
            match_id = 0
            for _, row in df_subset.iterrows():
                used_subj = [] if len(df_subset_matched.index)==0 else df_subset_matched['subj'].unique().tolist()
                
                best_match = None
                best_diff = time_diff_threshold
                
                for _, row_c in df_master.loc[df_master['cn_label']==1, ].iterrows():
                    if (row_c['subj'] in used_subj) or (row_c['sex'] != row['sex']):
                        continue
                    if (row_c['dataset'] != row['dataset']) and match_dataset:
                        continue
                    age_diff = abs(row_c['age'] - row['age'])
                    if age_diff < best_diff:
                        best_match = row_c
                        best_diff = age_diff
                
                if best_match is None:
                    for _, row_c in df_master.loc[df_master['cn_label']==0.5, ].iterrows():
                        if (row_c['subj'] in used_subj) or (row_c['sex'] != row['sex']):
                            continue
                        if (row_c['dataset'] != row['dataset']) and match_dataset:
                            continue
                        age_diff = abs(row_c['age'] - row['age'])
                        if age_diff < best_diff:
                            best_match = row_c
                            best_diff = age_diff
                
                if best_match is not None:
                    for r, clf_label in [(row, 1), (best_match, 0)]:
                        r = r.to_frame().T
                        r['clf_label'] = clf_label
                        r['match_id'] = match_id
                        df_subset_matched = pd.concat([df_subset_matched, r])
                    match_id += 1
                else:
                    print(f'No match found for {row["subj"]}')
        else:
            raise ValueError(f'Unknown mode: {mode}')
            
        return df_subset_matched

    def visualize_data_points(self, df, df_matched, png, disease='MCI', markout_matched_dp=True):
        """ Visualize i) the chronological age of longitudinal data points of CN subject, 
        ii) the chronological age and iii) time to AD/MCI of longitudinal data points of subjects 
        who have progressed from cognitively normal to MCI or AD.
            If markout_matched_dp is True, the data points that are matched with the CN data points will be marked out.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 12))
        
        # subplot-1: chronological age of CN subjects
        data = df.loc[df['cn_label']>=0.5, ].copy()
        data = data.sort_values(by='age')
        data['y_subject'] = None
        for i, subj in enumerate(data['subj'].unique()):
            data.loc[data['subj']==subj, 'y_subject'] = i
        sns.lineplot(
            data=data, x='age', y='y_subject', units='subj', estimator=None, 
            lw=1, color='tab:gray', alpha=0.5, linestyle='-', ax=axes[0]
        )
        
        cn_groups = {
            0.5: {'name': 'CN (without follow-up sessions)', 'color': 'greenyellow'},
            1: {'name': 'CN (with ≥1 follow-up CN session)', 'color': 'tab:green'},
        }
        for cn_label, group_props in cn_groups.items():
            axes[0].scatter(
                x=data.loc[data['cn_label']==cn_label, 'age'], y=data.loc[data['cn_label']==cn_label, 'y_subject'],
                s=5, c=group_props['color'], label=group_props['name']
            )
        axes[0].legend(loc='upper left')
        axes[0].set_ylabel(f'CN subjects (N={len(data["subj"].unique())}, sorted by baseline age)')
        axes[0].set_xlabel('chronological age (years)')

        # subplot-2: chronological age of subjects who progressed to MCI/AD (with/without matched CN data points)
        data = df.copy()
        data = data.sort_values(by='age')
        data['y_subject'] = None
        for i, subj in enumerate(data.loc[data[f'time_to_{disease}'].notna(), 'subj'].unique()):
            if markout_matched_dp:
                match_ids = df_matched.loc[df_matched['subj']==subj, 'match_id'].unique()
                for _, row in df_matched.loc[df_matched['match_id'].isin(match_ids), ].iterrows():
                    data.loc[(data['subj']==row['subj'])&(data['age']==row['age']), 'y_subject'] = i + 0.3
            data.loc[data['subj']==subj, 'y_subject'] = i
        data = data.loc[data['y_subject'].notna(), ].copy()

        sns.lineplot(
            data=data, x='age', y='y_subject', units='subj', estimator=None,
            lw=1, color='tab:gray', alpha=0.5, linestyle='-', ax=axes[1]
        )
        progression_phases = {
            'normal': {'name': 'CN (at present)', 'color': 'yellow'},
            'MCI': {'name': 'MCI', 'color': 'tab:orange'},
            'AD': {'name': 'AD', 'color': 'tab:red'},
        }
        for phase, props in progression_phases.items():
            axes[1].scatter(
                x=data.loc[data[f'time_to_{disease}'].notna()&(data['diagnosis']==phase), 'age'], 
                y=data.loc[data[f'time_to_{disease}'].notna()&(data['diagnosis']==phase), 'y_subject'],
                s=12, marker='^', c=props['color'], label=props['name']
            )
        axes[1].legend(loc='upper left')
        
        for cn_label, group_props in cn_groups.items():
            axes[1].scatter(
                x=data.loc[data['cn_label']==cn_label, 'age'], y=data.loc[data['cn_label']==cn_label, 'y_subject'],
                s=5, c=group_props['color'], label=group_props['name']
            )
        axes[1].set_ylabel(f'Subjects who progressed to {disease} from CN (N={len(data.loc[data[f"age_{disease}"].notna(), "subj"].unique())})')
        axes[1].set_xlabel('chronological age (years)')

        # subplot-3: time to MCI/AD of subjects who progressed to MCI/AD
        data = df.copy()
        data = data.sort_values(by='age')
        data['y_subject'] = None
        data_matched = df_matched.copy()
        data_matched['y_subject'] = None
        for i, subj in enumerate(data.loc[data[f'time_to_{disease}'].notna(), 'subj'].unique()):
            if markout_matched_dp:
                data_matched.loc[data_matched['subj']==subj, 'y_subject'] = i
            data.loc[data['subj']==subj, 'y_subject'] = i
        data = data.loc[data['y_subject'].notna(), ].copy()
        data_matched = data_matched.loc[data_matched['y_subject'].notna(), ].copy()
        
        sns.lineplot(
            data=data, x=f'time_to_{disease}', y='y_subject', units='subj', estimator=None,
            lw=1, color='tab:gray', alpha=0.5, linestyle='-', ax=axes[2]
        )
        for phase, props in progression_phases.items():
            axes[2].scatter(
                x=data.loc[data[f'time_to_{disease}'].notna()&(data['diagnosis']==phase), f'time_to_{disease}'], 
                y=data.loc[data[f'time_to_{disease}'].notna()&(data['diagnosis']==phase), 'y_subject'],
                s=12, marker='^', c=props['color']
            )
        if markout_matched_dp:
            axes[2].scatter(
                x=data_matched.loc[data_matched[f'time_to_{disease}'].notna(), f'time_to_{disease}'], 
                y=data_matched.loc[data_matched[f'time_to_{disease}'].notna(), 'y_subject'],
                s=16, marker='|', c='k', label='with matched CN data points'
            )
            axes[2].legend(loc='upper left')
            
        axes[2].set_xlabel(f'Time to {disease} (years)')
        axes[2].set_ylabel(f'Subjects who progressed to {disease} from CN (N={len(data.loc[data[f"age_{disease}"].notna(), "subj"].unique())})')
        axes[2].invert_xaxis()
        fig.savefig(png, dpi=300)


class LeaveOneSubjectOutDataLoader:
    """ Leave-one-subject-out data loader. 
    Given the dataframe (after CN data point matching), and the disease of interest,
    it will iterate through the subjects, each time returning the dataframe of the left-out subject (and matched CN data points)
    and the dataframe of all other subjects (and their matched CN data points)
    """
    def __init__(self, df, disease):
        self.df = df
        self.subjects_interest = df.loc[df[f'time_to_{disease}']>=0, 'subj'].unique()
        self.current_subject_index = 0
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self.subjects_interest)    
    
    def __next__(self):
        if self.current_subject_index < len(self.subjects_interest):
            subj = self.subjects_interest[self.current_subject_index]
            subj_match_id = self.df.loc[self.df['subj']==subj, 'match_id'].unique()
            df_left_out_subj = self.df.loc[self.df['match_id'].isin(subj_match_id), ]
            df_rest = self.df.loc[~self.df['match_id'].isin(subj_match_id), ]
            self.current_subject_index += 1
            return df_left_out_subj, df_rest
        else:
            raise StopIteration


def visualize_t_minus_n_prediction_results(df_aucs, dict_windowed_results, png):
    # hyperparameters
    fontsize = 9
    fontfamily = 'DejaVu Sans'
    linewidth = 2
    fig = plt.figure(figsize=(6.5, 8), tight_layout=True)
    gs = gridspec.GridSpec(nrows=4, ncols=3, wspace=0, hspace=0, width_ratios=[0.75, 0.25, 1],  height_ratios=[1, 1, 1, 1])
    clf_names = ['Logistic Regression','Linear SVM','Random Forest']
    dict_feat_combos = {
        'basic: chronological age + sex': {'color': (0,0,0), 'alpha': 0.5, 'linestyle': 'solid'},
        'basic + WM age nonlinear': {'color': (0,0,1), 'alpha': 1, 'linestyle': 'solid'},
        'basic + WM age affine': {'color': (0.5,0,1), 'alpha': 1, 'linestyle': 'solid'},
        'basic + GM age (ours)': {'color': (1,0,0), 'alpha': 1, 'linestyle': 'solid'},
        'basic + GM age (DeepBrainNet)': {'color': (1,0.5,0.5), 'alpha': 1, 'linestyle': 'dashed'},
        'basic + GM age (TSAN)': {'color': (0.5,0,0), 'alpha': 1, 'linestyle': 'dotted'},
        'basic + WM age nonlinear + GM age (ours)': {'color': (0,1,0), 'alpha': 1, 'linestyle': 'solid'},
        'basic + WM age nonlinear + GM age (DeepBrainNet)': {'color': (0.5,1,0.5), 'alpha': 1, 'linestyle': 'dashed'},        
        'basic + WM age nonlinear + GM age (TSAN)': {'color': (0,0.5,0), 'alpha': 1, 'linestyle': 'dotted'},        
    }
    feat_vis_order = [
        'basic: chronological age + sex',
        'basic + WM age nonlinear + GM age (TSAN)', 'basic + WM age nonlinear + GM age (DeepBrainNet)',
        'basic + WM age nonlinear + GM age (ours)',
        'basic + GM age (TSAN)', 'basic + GM age (DeepBrainNet)', 'basic + GM age (ours)',
        'basic + WM age affine', 'basic + WM age nonlinear',
    ]

    timetoevent_col = [col for col in df_aucs.columns if 'time_to_' in col]
    assert len(timetoevent_col) == 1
    timetoevent_col = timetoevent_col[0]
    disease = timetoevent_col.replace('time_to_', '')
    
    xlim= [-0.25, df_aucs[timetoevent_col].max()+0.5]
    ylim = [0.35, 0.85] if disease == 'MCI' else [0.35, 1]
    y_ticks = [0.4, 0.5, 0.6, 0.7, 0.8] if disease == 'MCI' else [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Upper left block: draw the legend
    ax = fig.add_subplot(gs[:3,0])
    lines = []
    for feat_combo in dict_feat_combos.keys():
        label_txt = textwrap.fill(feat_combo, width=25)
        line = mlines.Line2D([], [], color=dict_feat_combos[feat_combo]['color'], alpha=dict_feat_combos[feat_combo]['alpha'], linestyle=dict_feat_combos[feat_combo]['linestyle'], linewidth=linewidth, label=label_txt)
        lines.append(line)
    ax.legend(handles=lines, 
              prop={'size':fontsize, 'family':fontfamily}, 
              labelspacing=2.5, 
              frameon=True, 
              loc='upper left', 
              bbox_to_anchor=(0, 1), 
              title=f'Features for {disease} Prediction',
              title_fontproperties={'size':fontsize, 'family':fontfamily})
    ax.axis('off')
    
    # Upper middle block: y axis label
    ax = fig.add_subplot(gs[:3,1])
    ax.text(0.2, 0.5, 'Area under the Roc Curve', fontsize=fontsize, fontfamily=fontfamily, ha='center', va='center', rotation='vertical', transform=ax.transAxes)
    ax.axis('off')
    
    # Upper right block: draw the AUC plots
    for i, classifier in enumerate(clf_names):
        ax = fig.add_subplot(gs[i,2])
        for feat_combo in feat_vis_order:
            data = df_aucs.loc[(df_aucs['clf_name']==classifier)&(df_aucs['feat_combo_name']==feat_combo), ].copy()
            ax.plot(
                data[timetoevent_col].values,
                data['auc_mean'].values,
                linewidth=linewidth,
                linestyle=dict_feat_combos[feat_combo]['linestyle'],
                color=dict_feat_combos[feat_combo]['color'],
                alpha=dict_feat_combos[feat_combo]['alpha'],
                )
            ax.fill_between(
                x=data[timetoevent_col].values,
                y1=data['auc_upper'].values,
                y2=data['auc_lower'].values,
                color=dict_feat_combos[feat_combo]['color'],
                alpha=dict_feat_combos[feat_combo]['alpha']*0.1,
                linewidth=0)
            
        ax.vlines(x=df_aucs[timetoevent_col].unique(), ymin=0, ymax=1, transform=ax.get_xaxis_transform(), color=(0,0,0), linestyle='-', linewidth=1, alpha=0.1)
        ax.text(0.02, 0.95, classifier, fontsize=fontsize, fontfamily=fontfamily, transform=ax.transAxes, verticalalignment='top')
        ax.set_xlim(left=xlim[0], right=xlim[1])
        ax.set_xticks([])

        ax.set_ylim(bottom=ylim[0], top=ylim[1])
        ax.set_yticks(y_ticks)
        ax.invert_xaxis()
        ax.set_ylabel('')

    # Bottom middle block: y axis label
    ax = fig.add_subplot(gs[3,1])
    ax.text(0.2, 0.5, f'Subsets from Sliding Windows', fontsize=fontsize, fontfamily=fontfamily, ha='center', va='center', rotation='vertical', transform=ax.transAxes)
    ax.axis('off')
    
    # Bottom right: draw time-to-event distribution raincloud plot
    data_subsets = {'idx': [], timetoevent_col: [], 'num_pairs': []}
    for idx, w_results in dict_windowed_results.items():
        w_results = w_results.loc[w_results[timetoevent_col]>=0, ].copy()
        data_subsets['idx'] += [idx]*len(w_results.index)
        data_subsets[timetoevent_col] += w_results[timetoevent_col].values.tolist()
        data_subsets['num_pairs'] += [w_results['match_id'].nunique()] * len(w_results.index)
    data_subsets = pd.DataFrame(data_subsets)
    
    ax = fig.add_subplot(gs[3,2])
    sns.violinplot(data=data_subsets, x=timetoevent_col, y='idx', orient='h', color='gray', width=2, linewidth=1, split=True, inner=None, cut=0, density_norm='count', ax=ax)
    ax.vlines(x=df_aucs[timetoevent_col].unique(), ymin=0, ymax=1, transform=ax.get_xaxis_transform(), color=(0,0,0), linestyle='-', linewidth=1, alpha=0.1)
    for idx in data_subsets['idx'].unique():
        num = data_subsets.loc[data_subsets['idx']==idx, 'num_pairs'].values[0]
        ax.text(data_subsets.loc[data_subsets['idx']==idx, timetoevent_col].mean(), idx+1.5, f'{num}', fontsize=fontsize*0.75, fontfamily=fontfamily, ha='center', va='center')
    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.invert_xaxis()
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.set_xlabel(f"{timetoevent_col.replace('time_to_', 'Time to ')} (years)", fontsize=fontsize, fontfamily=fontfamily)
    
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=600)
    plt.close('all')
    
def user_input():
    parser = argparse.ArgumentParser(description='"T-0,T-1,...,T-N" MCI/AD prediction experiment')
    parser.add_argument('--wobc', action='store_true', help='when this flag is given, load predictions that are not bias-corrected')
    parser.add_argument('--disease', type=str, default='MCI', help='either "MCI" or "AD". Default: "MCI"')
    parser.add_argument('--match_mode', type=str, default='hungry_but_picky', help='either "hungry_but_picky", "hungry", "allow_multiple_per_subject", or "lavish". Default: "hungry_but_picky"')
    parser.add_argument('--match_dataset', action='store_true', help='when this flag is given, the data matching should consider the "dataset" label as well.')
    parser.add_argument('--age_min', type=int, default=0, help='the minimum age of data points to be included in the analysis. Default: 0 (years)')
    parser.add_argument('--age_max', type=int, default=1000, help='the maximum age of data points to be included in the analysis. Default: 1000 (years)')
    parser.add_argument('--window_size', type=int, default=2, help='the size of the sliding window. Default: 2 (years)')
    parser.add_argument('--run_all_exp', action='store_true', help='when this flag is given, run through all combinations of user inputs that make sense.')
    args = parser.parse_args()

    if args.run_all_exp:
        bias_correction_options = [True]
        disease_options = ['MCI', 'AD']
        match_mode_options = ['hungry_but_picky']
        match_dataset_options = [False]
        age_range_options = [(0, 1000), (45, 90)]
        window_size_options = [1, 2]
    else:
        bias_correction_options = [not args.wobc]
        disease_options = [args.disease]
        match_mode_options = [args.match_mode]
        match_dataset_options = [args.match_dataset]
        age_range_options = [(args.age_min, args.age_max)]
        window_size_options = [args.window_size]
    
    options = {
        'bias_correction': bias_correction_options,
        'disease': disease_options,
        'match_mode': match_mode_options,
        'match_dataset': match_dataset_options,
        'age_range': age_range_options,
        'window_size': window_size_options,
    }
    return options

def run_experiment(bias_correction, disease, match_mode, match_dataset, age_range, window_size):
    # Data Preparation
    data_prep = DataPreparation(roster_brain_age_models(), '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv')
    suffix = '_w-bc' if bias_correction else '_wo-bc'
    output_csv = f'experiments/2024-04-17_MCI_AD_Prediction_From_T_Minus_N_Years_Sliding_Window/data/data_prep{suffix}.csv'
    if Path(output_csv).is_file():
        df = pd.read_csv(output_csv)
    else:
        df = data_prep.load_predictions_of_all_models(bias_correction=bias_correction)
        df = data_prep.retrieve_diagnosis_label(df)
        df = data_prep.assign_cn_label(df)
        df = data_prep.feature_engineering(df)
        df = data_prep.mark_progression_subjects_out(df)
        df.to_csv(output_csv, index=False)
    
    # Data Matching
    suffix += f'_{disease}_age-{age_range[0]}-{age_range[1]}_{match_mode}_{"match-w-dataset" if match_dataset else "match-wo-dataset"}'
    output_csv = f'experiments/2024-04-17_MCI_AD_Prediction_From_T_Minus_N_Years_Sliding_Window/data/matched_dataset{suffix}.csv'
    if Path(output_csv).is_file():
        df_interest_matched = pd.read_csv(output_csv)
    else:
        df_interest = df.loc[(df[f'time_to_{disease}']>=0) & df['age'].between(age_range[0],age_range[1]), ].copy()
        df_interest_matched = data_prep.get_matched_cn_data(df_master=df, df_subset=df_interest, disease=disease, time_diff_threshold=1, mode=match_mode, match_dataset=match_dataset)
        df_interest_matched.to_csv(output_csv, index=False)
    data_prep.visualize_data_points(df, df_interest_matched, png=f'experiments/2024-04-17_MCI_AD_Prediction_From_T_Minus_N_Years_Sliding_Window/figs/vis_progression_data_points{suffix}.png', disease=disease)
    
    # Leave-one-subject-out prediction for each classifier-feature pair
    output_csv = f'experiments/2024-04-17_MCI_AD_Prediction_From_T_Minus_N_Years_Sliding_Window/data/prediction_results{suffix}.csv'
    if Path(output_csv).is_file():
        prediction_results = pd.read_csv(output_csv)
    else:
        classifiers = roster_classifiers()
        feat_combo = roster_feature_combinations(df)
        prediction_results = pd.DataFrame()

        for df_left_out_subj, df_rest in tqdm(LeaveOneSubjectOutDataLoader(df_interest_matched, disease), desc='Leave-one-subject-out prediction'):        
            for feat_combo_name, feat_cols in feat_combo.items():
                for clf_name, (clf, clf_params) in classifiers.items():
                    X_rest = df_rest[feat_cols].values
                    y_rest = df_rest['clf_label'].values
                    X_left_out = df_left_out_subj[feat_cols].values
                    y_left_out = df_left_out_subj['clf_label'].values
                    
                    # min-max normalization
                    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_rest)
                    X_rest = scaling.transform(X_rest)
                    X_left_out = scaling.transform(X_left_out)
                            
                    # impute missing values
                    imputer = SimpleImputer(strategy='mean')
                    imputer.fit(X_rest)
                    X_rest = imputer.transform(X_rest)
                    X_left_out = imputer.transform(X_left_out)

                    # classification
                    clf_instance = clf(**clf_params)
                    clf_instance.fit(X_rest, y_rest)
                    y_pred_proba = clf_instance.predict_proba(X_left_out)
                    
                    # record the prediction results
                    results = df_left_out_subj[['subj','sex','age','diagnosis','cn_label',f'age_{disease}',f'time_to_{disease}','clf_label','match_id']].copy()
                    results['feat_combo_name'] = feat_combo_name
                    results['clf_name'] = clf_name
                    results['y_pred_proba_0'] = y_pred_proba[:,0]
                    results['y_pred_proba_1'] = y_pred_proba[:,1]
                    prediction_results = pd.concat([prediction_results, results], ignore_index=True)
        prediction_results.to_csv(output_csv, index=False)
        
    # Sliding window
    window_step = window_size*0.5
    num_window = int((prediction_results[f'time_to_{disease}'].max() - window_size) / window_step) + 1
    dict_windowed_results = {}

    # T-0
    w_results = pd.DataFrame()
    for subj in prediction_results.loc[prediction_results[f'time_to_{disease}']==0, 'subj'].unique():
        subj_rows = prediction_results.loc[(prediction_results['subj']==subj)&(prediction_results[f'time_to_{disease}']==0), ]
        assert subj_rows['match_id'].nunique() == 1, f"Multiple data points found for {subj} at T-0."
        match_id = subj_rows['match_id'].unique()[0]
        w_results = pd.concat([w_results, prediction_results.loc[prediction_results['match_id']==match_id, ]])
    dict_windowed_results[0] = w_results
    
    # T-n
    idx = 1
    for i in range(num_window+1):
        window_start = i * window_step
        window_end = window_start + window_size
        
        # if no one is in the window, skip
        subjs = prediction_results.loc[(prediction_results[f'time_to_{disease}']>window_start)&(prediction_results[f'time_to_{disease}']<=window_end), 'subj'].unique()
        if len(subjs) < 3:
            print(f'Fewer than 3 subjects found in the window [{window_start}, {window_end}].')
            continue
        
        w_results = pd.DataFrame()
        for subj in subjs:
            subj_rows = prediction_results.loc[(prediction_results['subj']==subj)&(prediction_results[f'time_to_{disease}']>window_start)&(prediction_results[f'time_to_{disease}']<=window_end), ]
            
            if len(subj_rows['match_id'].unique()) == 1:
                match_id = subj_rows['match_id'].unique()[0]
            elif len(subj_rows['match_id'].unique()) > 1:
                most_center_match_id = None
                smallest_distance = window_size*0.5
                for match_id in subj_rows['match_id'].unique():
                    distance = abs(subj_rows.loc[subj_rows['match_id']==match_id, f'time_to_{disease}'].values[0] - (window_start + window_end)*0.5)
                    if distance < smallest_distance:
                        smallest_distance = distance
                        most_center_match_id = match_id
                match_id = most_center_match_id
            else:
                raise ValueError(f'No match found for {subj} in the window [{window_start}, {window_end}].')
            
            w_results = pd.concat([w_results, prediction_results.loc[prediction_results['match_id']==match_id, ]])
        dict_windowed_results[idx] = w_results
        idx += 1
        
    # AUC Bootstrap
    suffix += f'_ws-{window_size}'
    output_csv = f'experiments/2024-04-17_MCI_AD_Prediction_From_T_Minus_N_Years_Sliding_Window/data/prediction_auc_bootstrap{suffix}.csv'
    if Path(output_csv).is_file():
        df_aucs = pd.read_csv(output_csv)
    else:
        num_bootstrap = 1000  # number of bootstraps
        dict_aucs = {
            'idx': [], f'time_to_{disease}': [], 'feat_combo_name': [] , 'clf_name': [], 
            'auc_mean': [], 'auc_upper': [], 'auc_lower': []
            }

        for idx, w_results in tqdm(dict_windowed_results.items(), desc='AUC Bootstrap'):
            time_mean = w_results.loc[w_results['clf_label']==1, f'time_to_{disease}'].mean()
            
            for feat_combo_name in w_results['feat_combo_name'].unique():
                for clf_name in w_results['clf_name'].unique():
                    data = w_results.loc[(w_results['feat_combo_name']==feat_combo_name)&(w_results['clf_name']==clf_name), ].copy()
                    n_size = len(data['match_id'].unique())  # sample at pair-level
                    aucs = []
                                
                    for _ in range(num_bootstrap):
                        bootstrapped_ids = np.random.choice(data['match_id'].unique(), n_size, replace=True)
                        bootstrapped_data = data.loc[data['match_id'].isin(bootstrapped_ids), ]
                        auc = roc_auc_score(bootstrapped_data['clf_label'], bootstrapped_data['y_pred_proba_1'])
                        aucs.append(auc)
                        
                    auc_mean = np.mean(aucs)
                    confidence_interval = np.percentile(aucs, [2.5, 97.5])  # 95% CI
                    
                    dict_aucs['idx'].append(idx)
                    dict_aucs[f'time_to_{disease}'].append(time_mean)
                    dict_aucs['feat_combo_name'].append(feat_combo_name)
                    dict_aucs['clf_name'].append(clf_name)
                    dict_aucs['auc_mean'].append(auc_mean)
                    dict_aucs['auc_upper'].append(confidence_interval[1])
                    dict_aucs['auc_lower'].append(confidence_interval[0])
                    
        df_aucs = pd.DataFrame(dict_aucs)
        df_aucs.to_csv(output_csv, index=False)
    visualize_t_minus_n_prediction_results(df_aucs, dict_windowed_results, png=f'experiments/2024-04-17_MCI_AD_Prediction_From_T_Minus_N_Years_Sliding_Window/figs/vis_t_minus_n_prediction_results{suffix}.png')

if __name__ == '__main__':
    options = user_input()

    for bias_correction in options['bias_correction']:
        for disease in options['disease']:
            for match_mode in options['match_mode']:
                for match_dataset in options['match_dataset']:
                    for age_range in options['age_range']:
                        for window_size in options['window_size']:
                            run_experiment(bias_correction, disease, match_mode, match_dataset, age_range, window_size)