import pdb
import random
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from warnings import simplefilter
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
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
    return feat_combo


def roster_classifiers():
    classifiers = {
        'Logistic Regression': (LogisticRegression, {'random_state': 42, 'max_iter': 1000}),
        'Linear SVM': (SVC, {'kernel': "linear", 'C': 1, 'probability': True, 'random_state': 42}),
        'Random Forest': (RandomForestClassifier, {'max_depth': 5, 'n_estimators': 100, 'random_state': 42}),
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
        
        # remove duplicated rows (e.g., ses-270scanner09, ses-270scanner10 in BLSA)
        df = df.sort_values(by=['dataset', 'subject', 'age', 'session'], ignore_index=True)
        df = df.drop_duplicates(subset=['dataset', 'subject', 'age'], keep='last', ignore_index=True)
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
    
    def assign_cn_label(self, df):
        """ Create a new column "cn_label" to indicate whether the subject is cognitively normal.
        1:   the subject is cognitively normal, 
             and has only cognitively normal in his/her diagnosis history, 
             and has at least one following session in which the subject is still cognitively normal.
        0.5: the subject is cognitively normal,
             and has only cognitively normal in his/her diagnosis history.
        """
        if 'subj' not in df.columns:
            df['subj'] = df['dataset'] + '_' + df['subject']
        
        df['cn_label'] = None
        
        for subj in df.loc[df['diagnosis']=='normal', 'subj'].unique():
            if len(df.loc[df['subj']==subj, 'diagnosis'].unique())==1:  # there is only 'normal' in diagnosis history
                df.loc[df['subj']==subj, 'cn_label'] = 0.5
                if len(df.loc[df['subj']==subj, 'age'].unique())>=2:  # at least two sessions are available
                    # pick all but the last session (which is uncertain if it progresses to MCI/AD)
                    df.loc[(df['subj']==subj) & (df['age']!=df.loc[df['subj']==subj,'age'].max()), 'cn_label'] = 1
        num_subj_strict = len(df.loc[df['cn_label']==1, 'subj'].unique())
        num_subj_loose = len(df.loc[df['cn_label']>=0.5, 'subj'].unique())
        print(f'Found {num_subj_strict} subjects with strict CN label, and {num_subj_loose} subjects with loose CN label.')
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
            print(f'Found {num_subj} subjects with {disease} progression.')
        
        return df

    def visualize_data_points(self, df, png, disease='MCI', df_matched=None, markout_matched_dp=False):
        """ Visualize the chronological age, and time to AD/MCI of the data points 
        of subjects who have progressed from cognitively normal to MCI or AD.
            If markout_matched_dp is True, the data points which have been matched with CN data points
        will be marked out with a vertical line.
        """
        df = df.loc[df[f'age_{disease}'].notna(), ].copy()
        if 'subj' not in df.columns:
            df['subj'] = df['dataset'] + '_' + df['subject']
            
        df = df.sort_values(by='age')
        df['y_subject'] = None
        for i, subj in enumerate(df['subj'].unique()):
            df.loc[df['subj']==subj, 'y_subject'] = i
        
        if markout_matched_dp:
            df_matched = df_matched.loc[df_matched[f'time_to_{disease}'].notna(), ].copy()
            df_matched['y_subject'] = None
            for subj in df_matched['subj'].unique():
                df_matched.loc[df_matched['subj']==subj, 'y_subject'] = df.loc[df['subj']==subj, 'y_subject'].values[0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 12))
        
        for ax_id, x_axis in enumerate(['age', f'time_to_{disease}']):            
            sns.lineplot(
                data=df,
                x=x_axis, y='y_subject',
                units="subj",
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
            
            if markout_matched_dp:
                axes[ax_id].scatter(
                    df_matched[x_axis], df_matched['y_subject'],
                    c='k',
                    marker='|',
                    label='with matched CN data point'
                )
                axes[ax_id].legend()

            axes[ax_id].set_xlabel(f'{x_axis} (years)', fontsize=16, fontfamily='DejaVu Sans')
            axes[ax_id].set_ylabel('Subject', fontsize=16, fontfamily='DejaVu Sans')  
        axes[1].invert_xaxis()
        fig.savefig(png, dpi=300)
        
    def get_matched_cn_data(self, df_master, df_subset, age_diff_threshold=1, mode='hungry'):
        """ Use greedy algorithm to sample a subset of cognitively normal data points from the main dataframe df_master.
        The subset matches the data points in df_subset in terms of age and sex. The greedy algorithm will prioritize 
        using subjects that are cognitively normal under the strict definition (cn_label==1). If the search does not find 
        any data points that satisfy the age_diff_threshold, the greedy algorithm will then use the subjects that are 
        cognitively normal under the loose definition (cn_label==0.5).
            When mode==allow_multiple_per_subject, it is allowed to use more than one data points from the same subject 
        in df_master, as long as the data points are used to match the data points of the same subject in df_subset.
            When mode==hungry, the algorithm will try to match as many data points as possible. It will start from the 
        subjects in the df_subset that have the most data points to be matched, and use the subject in df_master that has
        the most matched data points to match with. It tends to prioritize using subjects that are cognitively normal under
        the strict definition. But unlike the "allow_multiple_per_subject" and "lavish" mode, it achieves such prioritization
        as a result of the data-intrinsic property, rather than by programmed rule.
            The matched subset will be concatenated with df_subset and returned. There will be two new columns:
            - "clf_label": 1 for disease, and 0 for cognitively normal.
            - "match_id": the ID of the matched pair. (could be useful for pair-level split)
        """
        if mode == 'hungry':
            df_candidates = df_master.loc[df_master['cn_label']>=0.5, ].copy()
            df_candidates['clf_label'] = 0
            df_candidates['match_id'] = None
            df_subset['clf_label'] = 1
            df_subset['match_id'] = None
            num_dp_total = len(df_subset.index)

            match_id = 0
            subj = None
            stop_ct = 0

            while stop_ct < 200:
                todo_subjects = df_subset.loc[df_subset['match_id'].isna(), ].groupby('subj').size().sort_values(ascending=False).index
                if len(todo_subjects) == 0:
                    print('All subjects matched.')
                    break
                elif len(todo_subjects)>10:
                    subj = todo_subjects[0] if subj != todo_subjects[0] else random.choice(todo_subjects[1:])
                else:
                    subj = random.choice(todo_subjects)
                
                subj_c_most_match = None
                subj_c_most_match_num = 0
                for subj_c in df_candidates['subj'].unique():
                    if subj_c in df_candidates.loc[df_candidates['match_id'].notna(), 'subj'].unique():
                        continue  # because already used in previous match
                    if df_candidates.loc[df_candidates['subj']==subj_c, 'sex'].values[0] != df_subset.loc[df_subset['subj']==subj, 'sex'].values[0]:
                        continue
                    
                    matched_age_c = []
                    for age in sorted(df_subset.loc[(df_subset['subj']==subj)&df_subset['match_id'].isna(), 'age'].unique()):
                        for age_c in sorted(df_candidates.loc[(df_candidates['subj']==subj_c)&df_candidates['match_id'].isna(), 'age'].unique()):
                            if (abs(age_c - age) < age_diff_threshold) and (age_c not in matched_age_c):
                                matched_age_c.append(age_c)
                                break

                    if len(matched_age_c) > subj_c_most_match_num:
                        subj_c_most_match = subj_c
                        subj_c_most_match_num = len(matched_age_c)
                
                if subj_c_most_match is not None:
                    ct_add = 0
                    matched_age_c = []
                    for age in sorted(df_subset.loc[(df_subset['subj']==subj)&df_subset['match_id'].isna(), 'age'].unique()):
                        for age_c in sorted(df_candidates.loc[(df_candidates['subj']==subj_c_most_match)&df_candidates['match_id'].isna(), 'age'].unique()):
                            if (abs(age_c - age) < age_diff_threshold) and (age_c not in matched_age_c):
                                df_subset.loc[(df_subset['subj']==subj)&(df_subset['age']==age), 'match_id'] = match_id
                                df_candidates.loc[(df_candidates['subj']==subj_c_most_match)&(df_candidates['age']==age_c), 'match_id'] = match_id
                                match_id += 1
                                ct_add += 1
                                break

                    assert ct_add == subj_c_most_match_num, f'ct_add: {ct_add}, subj_c_most_match_num: {subj_c_most_match_num}'
                    num_dp_matched = len(df_subset.loc[df_subset['match_id'].notna(), ].index)
                    print(f"Matched {subj_c_most_match_num} data points of {subj_c_most_match}. {num_dp_matched} (matched) / {num_dp_total} (total)")
                    stop_ct = 0
                else:
                    stop_ct += 1
                    print(f"Stop count: {stop_ct} / 200")
            df_subset_matched = pd.concat([
                df_subset.loc[df_subset['match_id'].notna(), ], 
                df_candidates.loc[df_candidates['match_id'].notna(), ]])
        
        elif mode == 'allow_multiple_per_subject':
            df_subset_matched = pd.DataFrame()
            match_id = 0
            for _, row in df_subset.iterrows():
                used_subj = [] if len(df_subset_matched.index)==0 else df_subset_matched['subj'].unique().tolist()
                
                best_match = None
                best_diff = age_diff_threshold
                
                for _, row_c in df_master.loc[df_master['cn_label']==1, ].iterrows():
                    if row_c['sex'] != row['sex']:
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
                        if row_c['sex'] != row['sex']:
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
                else:
                    print(f'No match found for {row["subj"]}')
        elif mode == 'lavish':
            df_subset_matched = pd.DataFrame()
            match_id = 0
            for _, row in df_subset.iterrows():
                used_subj = [] if len(df_subset_matched.index)==0 else df_subset_matched['subj'].unique().tolist()
                
                best_match = None
                best_diff = age_diff_threshold
                
                for _, row_c in df_master.loc[df_master['cn_label']==1, ].iterrows():
                    if (row_c['subj'] in used_subj) or (row_c['sex'] != row['sex']):
                        continue
                    age_diff = abs(row_c['age'] - row['age'])
                    if age_diff < best_diff:
                        best_match = row_c
                        best_diff = age_diff
                
                if best_match is None:
                    for _, row_c in df_master.loc[df_master['cn_label']==0.5, ].iterrows():
                        if (row_c['subj'] in used_subj) or (row_c['sex'] != row['sex']):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='"T-0,T-1,...,T-N" MCI/AD prediction experiment')
    parser.add_argument('--wobc', action='store_true', help='when this flag is given, load predictions that are not bias-corrected')
    parser.add_argument('--disease', type=str, default='MCI', help='either "MCI" or "AD"')
    parser.add_argument('--match_mode', type=str, default='hungry', help=(
        'either "hungry", "allow_multiple_per_subject", or "lavish". ' 
        '"hungry" uses the most greedy approach and thus will result in the most matched data points. '
        '"allow_multiple_per_subject" allows multiple data points from the same subject to be used for matching. '
        '"lavish" will use only one data point from each subject.'))
    args = parser.parse_args()
    DISEASE = args.disease
    BIAS_CORRECTION = not args.wobc

    # Data Preparation
    data_prep = DataPreparation(roster_brain_age_models(), '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv')
    df = data_prep.load_predictions_of_all_models(bias_correction=BIAS_CORRECTION)
    df = data_prep.retrieve_diagnosis_label(df)
    df = data_prep.assign_cn_label(df)
    df = data_prep.feature_engineering(df)
    df = data_prep.mark_progression_subjects_out(df)
    data_prep.visualize_data_points(df, f'experiments/2024-04-17_MCI_AD_Prediction_From_T_Minus_N_Years_Sliding_Window/figs/vis_progression_data_points_{DISEASE}.png', disease=DISEASE)
    df.to_csv('experiments/2024-04-17_MCI_AD_Prediction_From_T_Minus_N_Years_Sliding_Window/data/data_prep.csv', index=False)
    df = pd.read_csv('experiments/2024-04-17_MCI_AD_Prediction_From_T_Minus_N_Years_Sliding_Window/data/data_prep.csv')

    # Data Matching
    df_interest = df.loc[df[f'time_to_{DISEASE}']>=0, ].copy()
    df_interest_matched = data_prep.get_matched_cn_data(df_master=df, df_subset=df_interest, age_diff_threshold=1, mode=args.match_mode)
    df_interest_matched.to_csv(f'experiments/2024-04-17_MCI_AD_Prediction_From_T_Minus_N_Years_Sliding_Window/data/matched_dataset_{DISEASE}_{args.match_mode}.csv', index=False)
    df_interest_matched = pd.read_csv(f'experiments/2024-04-17_MCI_AD_Prediction_From_T_Minus_N_Years_Sliding_Window/data/matched_dataset_{DISEASE}_{args.match_mode}.csv')
    data_prep.visualize_data_points(df, f'experiments/2024-04-17_MCI_AD_Prediction_From_T_Minus_N_Years_Sliding_Window/figs/vis_progression_data_points_{DISEASE}_matched_with_{args.match_mode}.png', disease=DISEASE, df_matched=df_interest_matched, markout_matched_dp=True)
    
    # Leave-one-subject-out prediction for each classifier-feature pair
    classifiers = roster_classifiers()
    feat_combo = roster_feature_combinations(df)
    prediction_results = pd.DataFrame()

    for df_left_out_subj, df_rest in tqdm(LeaveOneSubjectOutDataLoader(df_interest_matched, DISEASE), desc='Leave-one-subject-out prediction'):
        for feat_combo_name, feat_cols in feat_combo.items():
            for clf_name, (clf, clf_params) in classifiers.items():
                X = df_rest[feat_cols].values
                y = df_rest['clf_label'].values
                X_left_out = df_left_out_subj[feat_cols].values
                y_left_out = df_left_out_subj['clf_label'].values
                
                # min-max normalization
                scaling = MinMaxScaler(feature_range=(-1,1)).fit(X)
                X = scaling.transform(X)
                X_left_out = scaling.transform(X_left_out)
                        
                # impute missing values
                imputer = SimpleImputer(strategy='mean')
                imputer.fit(X)
                X = imputer.transform(X)
                X_left_out = imputer.transform(X_left_out)

                # classification
                clf_instance = clf(**clf_params)
                clf_instance.fit(X, y)
                y_pred_proba = clf_instance.predict_proba(X_left_out)
                
                # record the prediction results
                results = df_left_out_subj[['subj','sex','age','diagnosis','cn_label',f'age_{DISEASE}',f'time_to_{DISEASE}','clf_label','match_id']].copy()
                results['feat_combo_name'] = feat_combo_name
                results['clf_name'] = clf_name
                results['y_pred_proba_0'] = y_pred_proba[:,0]
                results['y_pred_proba_1'] = y_pred_proba[:,1]
                prediction_results = pd.concat([prediction_results, results], ignore_index=True)
    prediction_results.to_csv(f'experiments/2024-04-17_MCI_AD_Prediction_From_T_Minus_N_Years_Sliding_Window/data/prediction_results_{DISEASE}_{args.match_mode}.csv', index=False)
    
    # # Sliding window and AUC bootstrap
    # window_size = 1
    # window_step = 0.5

    # dict_windowed_results = {0: prediction_results.loc[prediction_results[f'time_to_{DISEASE}']==0, ].copy()}
    