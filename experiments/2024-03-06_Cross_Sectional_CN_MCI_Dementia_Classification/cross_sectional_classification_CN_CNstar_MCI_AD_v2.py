"""
New features:
- automatic feature selection
"""

import pdb
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score

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
        df['subj'] = df['dataset'] + '_' + df['subject']
        # first session of the disease-free subjects (with at least one follow-up session)
        for subj in df.loc[df['diagnosis']=='normal', 'subj'].unique():
            if (len(df.loc[df['subj']==subj, 'diagnosis'].unique()) == 1) and (len(df.loc[df['subj']==subj, 'age_gt'].unique()) >= 2):
                df.loc[(df['subj'] == subj) & (df['age_gt'] == df.loc[df['subj'] == subj, 'age_gt'].min()), 'category'] = 'CN'
        
        # session after which the subject converted to MCI or dementia
        for subj in df.loc[df['diagnosis'].isin(['MCI', 'dementia']), 'subj'].unique():
            if subj in df.loc[df['category'].notna(), 'subj'].unique():
                continue
            rows_subj = df.loc[df['subj']==subj, ].copy()
            if 'normal' in rows_subj['diagnosis'].values:
                rows_subj = rows_subj.sort_values(by='age_gt')
                for i in range(len(rows_subj.index)-1):
                    if (rows_subj.iloc[i]['diagnosis'] == 'normal') & (rows_subj.iloc[i+1]['diagnosis'] in ['MCI', 'dementia']):
                        df.loc[(df['subj'] == subj) & (df['age_gt'] == rows_subj.iloc[i]['age_gt']), 'category'] = 'CN*'
                        break
        
        # the last session of the MCI subjects
        for subj in df.loc[df['diagnosis']=='MCI', 'subj'].unique():
            if subj in df.loc[df['category'].notna(), 'subj'].unique():
                continue
            df.loc[(df['subj'] == subj) &
                   (df['age_gt'] == df.loc[(df['subj'] == subj)&(df['diagnosis']=='MCI'), 'age_gt'].max()), 'category'] = 'MCI'
        
        # the last session of the dementia subjects
        for subj in df.loc[df['diagnosis']=='dementia', 'subj'].unique():
            if subj in df.loc[df['category'].notna(), 'subj'].unique():
                continue
            df.loc[(df['subj'] == subj) &
                   (df['age_gt'] == df.loc[(df['subj'] == subj)&(df['diagnosis']=='dementia'), 'age_gt'].max()), 'category'] = 'AD'
        
        df = df.dropna(subset=['category'])
        return df
        
    def over_sampling(self, df, column_class='category', random_state=0):
        """ over-sampling the minority classes to balance the dataset """
        num_max = df[column_class].value_counts().max()
        for c in df[column_class].unique():
            num_ori = df[column_class].value_counts()[c]
            num_add = num_max - num_ori
            if num_add == 0:
                continue
            elif num_add <= num_ori:
                new_samples = df.loc[df[column_class]==c, ].sample(n=num_add, replace=False, random_state=random_state)
                df = pd.concat([df, new_samples], axis=0)
            elif num_add > num_ori:
                new_samples = df.loc[df[column_class]==c, ].sample(n=num_add, replace=True, random_state=random_state)
                df = pd.concat([df, new_samples], axis=0)
        return df

    def subject_level_splitting(self, df, num_folds=5, column_name='fold_idx', random_state=0):
        """ split the dataset at the subject level for cross-validation,
        save the fold information in a seperate column
        """
        assert column_name not in df.columns, f'Column {column_name} already exists in the dataframe'

        df[column_name] = None

        for c in df['category'].unique():
            subj_category = df.loc[df['category']==c, 'subj'].unique().tolist()
            random.seed(random_state)
            random.shuffle(subj_category)
            indices = [int(i*len(subj_category)/num_folds) for i in range(num_folds+1)]
            
            for i in range(num_folds):
                df.loc[(df['category']==c) & df['subj'].isin(subj_category[indices[i]:indices[i+1]]), column_name] = i+1
        
        assert df[column_name].notna().all(), f'Not all samples are assigned to a fold'                
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

# d = DataPreparation(dict_models, databank_csv='/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv')
# df = d.load_data()
# df = d.take_cross_sectional_samples(df)
# df = d.over_sampling(df)
# df = d.subject_level_splitting(df)
# df.to_csv('experiments/2024-03-06_Cross_Sectional_CN_MCI_Dementia_Classification/data.csv', index=False)
df = pd.read_csv('experiments/2024-03-06_Cross_Sectional_CN_MCI_Dementia_Classification/data.csv')

# feature combinations
feat_combo = {}
feat_combo['chronological age only'] = ['age_gt']
feat_combo['WM age only'] = ['age_pred_wm_age_mean']
feat_combo['WM age (contaminated) only'] = ['age_pred_wm_age_contaminated_mean']
feat_combo['GM age only'] = ['age_pred_gm_age_ours_mean']
feat_combo['GM age (TSAN) only'] = ['age_pred_gm_age_tsan_mean']
feat_combo['chronological + WM age (each fold)'] = ['age_gt'] + [f'age_pred_wm_age_{i}' for i in [1,2,3,4,5]]
feat_combo['chronological + WM age (mean)'] = ['age_gt'] + ['age_pred_wm_age_mean']
feat_combo['chronological + WM age (each fold + mean)'] = ['age_gt'] + [f'age_pred_wm_age_{i}' for i in [1,2,3,4,5]] + ['age_pred_wm_age_mean']
feat_combo['chronological + GM age (each fold)'] = ['age_gt'] + [f'age_pred_gm_age_ours_{i}' for i in [1,2,3,4,5]]
feat_combo['chronological + GM age (mean)'] = ['age_gt'] + ['age_pred_gm_age_ours_mean']
feat_combo['chronological + GM age (each fold + mean)'] = ['age_gt'] + [f'age_pred_gm_age_ours_{i}' for i in [1,2,3,4,5]] + ['age_pred_gm_age_ours_mean']
feat_combo['chronological + WM age (contaminated) (each fold)'] = ['age_gt'] + [f'age_pred_wm_age_contaminated_{i}' for i in [1,2,3,4,5]]
feat_combo['chronological + WM age (contaminated) (mean)'] = ['age_gt'] + ['age_pred_wm_age_contaminated_mean']
feat_combo['chronological + WM age (contaminated) (each fold + mean)'] = ['age_gt'] + [f'age_pred_wm_age_contaminated_{i}' for i in [1,2,3,4,5]] + ['age_pred_wm_age_contaminated_mean']
feat_combo['chronological + WM age + GM age (each fold)'] = ['age_gt'] + [f'age_pred_wm_age_{i}' for i in [1,2,3,4,5]] + [f'age_pred_gm_age_ours_{i}' for i in [1,2,3,4,5]]
feat_combo['chronological + WM age + GM age (mean)'] = ['age_gt'] + ['age_pred_wm_age_mean'] + ['age_pred_gm_age_ours_mean']
feat_combo['chronological + WM age + GM age (each fold + mean)'] = ['age_gt'] + [f'age_pred_wm_age_{i}' for i in [1,2,3,4,5]] + ['age_pred_wm_age_mean'] + [f'age_pred_gm_age_ours_{i}' for i in [1,2,3,4,5]] + ['age_pred_gm_age_ours_mean']
feat_combo['chronological + GM age (TSAN) (each fold)'] = ['age_gt'] + [f'age_pred_gm_age_tsan_{i}' for i in [1,2,3,4,5]]
feat_combo['chronological + GM age (TSAN) (mean)'] = ['age_gt'] + ['age_pred_gm_age_tsan_mean']
feat_combo['chronological + GM age (TSAN) (each fold + mean)'] = ['age_gt'] + [f'age_pred_gm_age_tsan_{i}' for i in [1,2,3,4,5]] + ['age_pred_gm_age_tsan_mean']

# classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Linear SVM': SVC(kernel="linear", C=0.025, probability=True, random_state=42),
    'RBF SVM': SVC(gamma=2, C=1, probability=True, random_state=42),
    'Gaussian Process': GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    'Naive Bayes': GaussianNB(),
    'Nearest Neighbors': KNeighborsClassifier(3),
}

# Task 1: CN vs AD
data = df.loc[df['category'].isin(['CN', 'AD']), ].copy()
data['category'] = data['category'].map({'CN': 0, 'AD': 1})

results = pd.DataFrame()
for combo_name, features in tqdm(feat_combo.items(), total=len(feat_combo), desc='Classification'):
    num_features = len(features)
    for classifier_name, clf in classifiers.items():
        accs, baccs, specs, senss, aucs = [], [], [], [], []
        for fold_idx in [1,2,3,4,5]:
            X_train = data.loc[data['fold_idx']!=fold_idx, features].values
            y_train = data.loc[data['fold_idx']!=fold_idx, 'category'].values
            X_test = data.loc[data['fold_idx']==fold_idx, features].values
            y_test = data.loc[data['fold_idx']==fold_idx, 'category'].values
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            bacc = balanced_accuracy_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            spec = tn / (tn + fp)
            sens = tp / (tp + fn)
            auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
            accs.append(acc)
            baccs.append(bacc)
            specs.append(spec)
            senss.append(sens)
            aucs.append(auc)

        acc_mean = sum(accs) / len(accs)
        acc_std = (sum([(x - acc_mean)**2 for x in accs])/len(accs))**0.5
        bacc_mean = sum(baccs) / len(baccs)
        bacc_std = (sum([(x - bacc_mean)**2 for x in baccs])/len(baccs))**0.5
        spec_mean = sum(specs) / len(specs)
        spec_std = (sum([(x - spec_mean)**2 for x in specs])/len(specs))**0.5
        sens_mean = sum(senss) / len(senss)
        sens_std = (sum([(x - sens_mean)**2 for x in senss])/len(senss))**0.5
        auc_mean = sum(aucs) / len(aucs)
        auc_std = (sum([(x - auc_mean)**2 for x in aucs])/len(aucs))**0.5

        result = pd.DataFrame({
            'feature_combination': [combo_name],
            'num_features': [num_features],
            'classifier': [classifier_name],
            'acc_mean': [acc_mean],
            'acc_std': [acc_std],
            'bacc_mean': [bacc_mean],
            'bacc_std': [bacc_std],
            'spec_mean': [spec_mean],
            'spec_std': [spec_std],
            'sens_mean': [sens_mean],
            'sens_std': [sens_std],
            'auc_mean': [auc_mean],
            'auc_std': [auc_std],
            'sanity_check_acc': [str(accs)],
            'sanity_check_bacc': [str(baccs)],
            'sanity_check_spec': [str(specs)],
            'sanity_check_sens': [str(senss)],
            'sanity_check_auc': [str(aucs)]
        })
        results = pd.concat([results, result], axis=0)
results.to_csv('experiments/2024-03-06_Cross_Sectional_CN_MCI_Dementia_Classification/results_CN_vs_AD.csv', index=False)

# Task 2: CN vs MCI
data = df.loc[df['category'].isin(['CN', 'MCI']), ].copy()
data['category'] = data['category'].map({'CN': 0, 'MCI': 1})

results = pd.DataFrame()
for combo_name, features in tqdm(feat_combo.items(), total=len(feat_combo), desc='Classification'):
    num_features = len(features)
    for classifier_name, clf in classifiers.items():
        accs, baccs, specs, senss, aucs = [], [], [], [], []
        for fold_idx in [1,2,3,4,5]:
            X_train = data.loc[data['fold_idx']!=fold_idx, features].values
            y_train = data.loc[data['fold_idx']!=fold_idx, 'category'].values
            X_test = data.loc[data['fold_idx']==fold_idx, features].values
            y_test = data.loc[data['fold_idx']==fold_idx, 'category'].values
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            bacc = balanced_accuracy_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            spec = tn / (tn + fp)
            sens = tp / (tp + fn)
            auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
            accs.append(acc)
            baccs.append(bacc)
            specs.append(spec)
            senss.append(sens)
            aucs.append(auc)

        acc_mean = sum(accs) / len(accs)
        acc_std = (sum([(x - acc_mean)**2 for x in accs])/len(accs))**0.5
        bacc_mean = sum(baccs) / len(baccs)
        bacc_std = (sum([(x - bacc_mean)**2 for x in baccs])/len(baccs))**0.5
        spec_mean = sum(specs) / len(specs)
        spec_std = (sum([(x - spec_mean)**2 for x in specs])/len(specs))**0.5
        sens_mean = sum(senss) / len(senss)
        sens_std = (sum([(x - sens_mean)**2 for x in senss])/len(senss))**0.5
        auc_mean = sum(aucs) / len(aucs)
        auc_std = (sum([(x - auc_mean)**2 for x in aucs])/len(aucs))**0.5

        result = pd.DataFrame({
            'feature_combination': [combo_name],
            'num_features': [num_features],
            'classifier': [classifier_name],
            'acc_mean': [acc_mean],
            'acc_std': [acc_std],
            'bacc_mean': [bacc_mean],
            'bacc_std': [bacc_std],
            'spec_mean': [spec_mean],
            'spec_std': [spec_std],
            'sens_mean': [sens_mean],
            'sens_std': [sens_std],
            'auc_mean': [auc_mean],
            'auc_std': [auc_std],
            'sanity_check_acc': [str(accs)],
            'sanity_check_bacc': [str(baccs)],
            'sanity_check_spec': [str(specs)],
            'sanity_check_sens': [str(senss)],
            'sanity_check_auc': [str(aucs)]
        })
        results = pd.concat([results, result], axis=0)
results.to_csv('experiments/2024-03-06_Cross_Sectional_CN_MCI_Dementia_Classification/results_CN_vs_MCI.csv', index=False)

# Task 3: CN vs CN*
data = df.loc[df['category'].isin(['CN', 'CN*']), ].copy()
data['category'] = data['category'].map({'CN': 0, 'CN*': 1})

results = pd.DataFrame()
for combo_name, features in tqdm(feat_combo.items(), total=len(feat_combo), desc='Classification'):
    num_features = len(features)
    for classifier_name, clf in classifiers.items():
        accs, baccs, specs, senss, aucs = [], [], [], [], []
        for fold_idx in [1,2,3,4,5]:
            X_train = data.loc[data['fold_idx']!=fold_idx, features].values
            y_train = data.loc[data['fold_idx']!=fold_idx, 'category'].values
            X_test = data.loc[data['fold_idx']==fold_idx, features].values
            y_test = data.loc[data['fold_idx']==fold_idx, 'category'].values
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            bacc = balanced_accuracy_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            spec = tn / (tn + fp)
            sens = tp / (tp + fn)
            auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
            accs.append(acc)
            baccs.append(bacc)
            specs.append(spec)
            senss.append(sens)
            aucs.append(auc)

        acc_mean = sum(accs) / len(accs)
        acc_std = (sum([(x - acc_mean)**2 for x in accs])/len(accs))**0.5
        bacc_mean = sum(baccs) / len(baccs)
        bacc_std = (sum([(x - bacc_mean)**2 for x in baccs])/len(baccs))**0.5
        spec_mean = sum(specs) / len(specs)
        spec_std = (sum([(x - spec_mean)**2 for x in specs])/len(specs))**0.5
        sens_mean = sum(senss) / len(senss)
        sens_std = (sum([(x - sens_mean)**2 for x in senss])/len(senss))**0.5
        auc_mean = sum(aucs) / len(aucs)
        auc_std = (sum([(x - auc_mean)**2 for x in aucs])/len(aucs))**0.5

        result = pd.DataFrame({
            'feature_combination': [combo_name],
            'num_features': [num_features],
            'classifier': [classifier_name],
            'acc_mean': [acc_mean],
            'acc_std': [acc_std],
            'bacc_mean': [bacc_mean],
            'bacc_std': [bacc_std],
            'spec_mean': [spec_mean],
            'spec_std': [spec_std],
            'sens_mean': [sens_mean],
            'sens_std': [sens_std],
            'auc_mean': [auc_mean],
            'auc_std': [auc_std],
            'sanity_check_acc': [str(accs)],
            'sanity_check_bacc': [str(baccs)],
            'sanity_check_spec': [str(specs)],
            'sanity_check_sens': [str(senss)],
            'sanity_check_auc': [str(aucs)]
        })
        results = pd.concat([results, result], axis=0)
results.to_csv('experiments/2024-03-06_Cross_Sectional_CN_MCI_Dementia_Classification/results_CN_vs_CNstar.csv', index=False)
