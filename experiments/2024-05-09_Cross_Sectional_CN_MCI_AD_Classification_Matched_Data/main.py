import pdb
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from warnings import simplefilter
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
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
        
        # Impute NaN values with the value from the closest session (if there is any) of the subject
        cols = [col for col in df.columns if '_bag_change_rate' in col]
        for subj in df['subj'].unique():
            ages = sorted(df.loc[df['subj']==subj, 'age'].unique())
            if len(ages) == 1:
                continue
            for col in cols:
                df.loc[(df['subj']==subj)&(df['age']==ages[-1]), col] = df.loc[(df['subj']==subj)&(df['age']==ages[-2]), col].values[0]            
        
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


def get_three_matched_datasets(df, cnstar_threshold=(0,6), age_diff_threshold=1):
    # Assign category labels
    df['category'] = None
    df.loc[df['cn_label']>=0.5, 'category'] = 'CN'
    df.loc[(df['time_to_MCI']>cnstar_threshold[0])&(df['time_to_MCI']<=cnstar_threshold[1]), 'category'] = 'CN*'
    df.loc[df['diagnosis']=='MCI', 'category'] = 'MCI'
    df.loc[df['diagnosis']=='AD', 'category'] = 'AD'
    
    # TODO: apply age filter here
    # df = df.loc[df['age'].between(45, 90), ].copy()
    
    dict_datasets = {
        'CN vs. AD': {'match_order': ['AD', 'CN'], 'matched_data': None}, 
        'CN vs. MCI': {'match_order': ['MCI', 'CN'], 'matched_data': None},
        'CN vs. CN*': {'match_order': ['CN*', 'CN'], 'matched_data': None},
    }
    
    for task_name in dict_datasets.keys():
        match_order = dict_datasets[task_name]['match_order']
        dfs_pool = {c: df.loc[df['category']==c, ].copy() for c in match_order}
        dfs_matched = {c: pd.DataFrame() for c in match_order}
        match_id = 0
        
        for _, row in tqdm(dfs_pool[match_order[0]].iterrows(), total=len(dfs_pool[match_order[0]].index), desc=f'Matching data for {task_name}'):
            used_subjs = []
            for c in match_order:
                if len(dfs_matched[c].index) == 0:
                    continue
                else:
                    used_subjs += dfs_matched[c]['subj'].unique().tolist()

            if row['subj'] in used_subjs:
                continue

            tmp_matched = {c: None for c in match_order[1:]}
            for c in match_order[1:]:
                smallest_age_diff = age_diff_threshold
                for j, row_c in dfs_pool[c].iterrows():
                    if (row_c['subj'] in used_subjs) or (row_c['sex']!=row['sex']):  # already used or different sex
                        continue
                    if (c=='CN') and (match_order[0]=='CN*') and (row_c['time_to_last_cn'] < row['time_to_MCI'] - age_diff_threshold):
                        continue
                    # In the future, if we have more data points, we can apply the following criteria:
                    # if row_c['dataset'] != row['dataset']:
                    #     continue
                    
                    age_diff = abs(row['age'] - row_c['age'])
                    if age_diff < smallest_age_diff:
                        smallest_age_diff = age_diff
                        tmp_matched[c] = row_c
                if tmp_matched[c] is not None:
                    used_subjs.append(tmp_matched[c]['subj'])
            
            if all([tmp_matched[c] is not None for c in match_order[1:]]):  # this sample has been matched in all categories
                for c in match_order:
                    r = row.to_frame().T if c==match_order[0] else tmp_matched[c].to_frame().T
                    r['match_id'] = match_id
                    dfs_matched[c] = pd.concat([dfs_matched[c], r])                
                match_id += 1
        print(f'--------> Greedy matching completed. Found {match_id} pairs.')
        
        # sanity check
        for i, c in enumerate(match_order):
            for j in range(i, len(match_order)):
                if i == j:
                    # make sure that no repeated subjects are in the same category
                    assert len(dfs_matched[c]['subj'].unique()) == len(dfs_matched[c].index), f"Repeated subjects in {c}"
                else:
                    # no overlapping subjects
                    assert len(set(dfs_matched[c]['subj'].unique()).intersection(set(dfs_matched[match_order[j]]['subj'].unique()))) == 0, f"Overlapping subjects between {c} and {match_order[j]}"
                    
        # merge into one dataframe
        df_matched = pd.DataFrame()
        for c in match_order:
            df_matched = pd.concat([df_matched, dfs_matched[c]], ignore_index=True)
        df_matched = df_matched.sort_values(by=['match_id', 'category'], ignore_index=True)
        dict_datasets[task_name]['matched_data'] = df_matched.copy()
    
    return dict_datasets
        

class LeaveOneOutDataLoader:
    def __init__(self, df):
        self.df = df
        self.match_ids = self.df['match_id'].unique()
        self.i = 0
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self.match_ids)
    
    def __next__(self):
        if self.i < len(self.match_ids):
            df_leftout = self.df.loc[self.df['match_id']==self.match_ids[self.i], ].copy()
            df_rest = self.df.loc[self.df['match_id']!=self.match_ids[self.i], ].copy()
            self.i += 1
            return df_leftout, df_rest
        else:
            raise StopIteration
    
    
def run_classification_experiments(df_matched, classifiers, feat_combo, savedir, num_bootstrap=1000):
    categories = df_matched['category'].unique()
    assert len(categories) == 2, "Only binary classification is supported."
    
    df_matched['clf_label'] = df_matched['category'].map({categories[0]: 0, categories[1]: 1})  # note that disease group could be 0
    
    # leave-one-out cross validation, store predicted probabilities
    results = pd.DataFrame()
    for df_leftout, df_rest in tqdm(LeaveOneOutDataLoader(df_matched), desc='LOO prediction'):
        for feat_combo_name, feat_cols in feat_combo.items():
            for clf_name, (clf, clf_params) in classifiers.items():    
                X_rest = df_rest[feat_cols].values
                y_rest = df_rest['clf_label'].values
                X_left_out = df_leftout[feat_cols].values
                y_left_out = df_leftout['clf_label'].values
                        
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
                y_pred_label = clf_instance.predict(X_left_out)
                y_pred_proba = clf_instance.predict_proba(X_left_out)[:,1]
                
                # record the prediction results
                res = df_leftout[['match_id','subj','age','category','clf_label']].copy()
                res['feat_combo_name'] = feat_combo_name
                res['clf_name'] = clf_name
                res['y_pred_label'] = y_pred_label
                res['y_pred_proba'] = y_pred_proba
                results = pd.concat([results, res], ignore_index=True)
    csv = Path(savedir) / f'predictions_{categories[0]}_vs_{categories[1]}.csv'
    results.to_csv(csv, index=False)
    
    # Bootstrap
    report = {
        'feat_combo_name': [], 'clf_name': [],
        'acc_mean': [], 'acc_upper': [], 'acc_lower': [],
        'auc_mean': [], 'auc_upper': [], 'auc_lower': [],
    }
        
    for feat_combo_name in results['feat_combo_name'].unique():
        for clf_name in results['clf_name'].unique():
            data = results.loc[(results['feat_combo_name']==feat_combo_name)&(results['clf_name']==clf_name), ].copy()
            n_size = data['match_id'].nunique()  # sample at pair-level
            accs = []
            aucs = []
                        
            for _ in range(num_bootstrap):
                bootstrapped_ids = np.random.choice(data['match_id'].unique(), n_size, replace=True)
                bootstrapped_data = data.loc[data['match_id'].isin(bootstrapped_ids), ]
                acc = accuracy_score(bootstrapped_data['clf_label'], bootstrapped_data['y_pred_label'])
                auc = roc_auc_score(bootstrapped_data['clf_label'], bootstrapped_data['y_pred_proba'])
                accs.append(acc)
                aucs.append(auc)
                
            acc_mean = np.mean(accs)
            acc_ci = np.percentile(accs, [2.5, 97.5])            
            auc_mean = np.mean(aucs)
            auc_ci = np.percentile(aucs, [2.5, 97.5])
            report['feat_combo_name'].append(feat_combo_name)
            report['clf_name'].append(clf_name)
            report['acc_mean'].append(acc_mean)
            report['acc_upper'].append(acc_ci[1])
            report['acc_lower'].append(acc_ci[0])
            report['auc_mean'].append(auc_mean)
            report['auc_upper'].append(auc_ci[1])
            report['auc_lower'].append(auc_ci[0])
    df_report = pd.DataFrame(report)
    csv = Path(savedir) / f'report_{categories[0]}_vs_{categories[1]}.csv'
    df_report.to_csv(csv, index=False)


if __name__ == '__main__':
    # Load data
    output_fn= 'experiments/2024-05-09_Cross_Sectional_CN_MCI_AD_Classification_Matched_Data/data/data_prep.csv'
    if Path(output_fn).is_file():
        df = pd.read_csv(output_fn)
    else:
        data_prep = DataPreparation(roster_brain_age_models(), '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv')
        df = data_prep.load_predictions_of_all_models(bias_correction=True)
        df = data_prep.retrieve_diagnosis_label(df)
        df = data_prep.assign_cn_label(df)
        df = data_prep.feature_engineering(df)
        df = data_prep.mark_progression_subjects_out(df)
        df.to_csv(output_fn, index=False)
        
    # Get matched data for three tasks: CN vs. AD, CN vs. MCI, CN vs. CN*
    output_fn = 'experiments/2024-05-09_Cross_Sectional_CN_MCI_AD_Classification_Matched_Data/data/matched_datasets.pkl'
    if Path(output_fn).is_file():
        with open(output_fn, 'rb') as f:
            dict_datasets = pickle.load(f)
    else:
        dict_datasets = get_three_matched_datasets(df, cnstar_threshold=(0,6), age_diff_threshold=1)
        with open(output_fn, 'wb') as f:
            pickle.dump(dict_datasets, f)
    
    # Classification
    for _, task_data in dict_datasets.items():
        run_classification_experiments(
            df_matched = task_data['matched_data'], 
            classifiers = roster_classifiers(), 
            feat_combo = roster_feature_combinations(df), 
            savedir = 'experiments/2024-05-09_Cross_Sectional_CN_MCI_AD_Classification_Matched_Data/data', 
            num_bootstrap=1000)