""" New features:
- Include brain age gap (BAG) and its interaction with chronological age as candidate features
"""

import pdb
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score


def run_classification_experiments(data, feat_combo, classifiers, results_csv):
    results = pd.DataFrame()
    results_simple = pd.DataFrame()
    for combo_name, list_feat in feat_combo.items():
        row = {'Features': [combo_name]}
        row_simple = {'Features': [combo_name]}
        for classifier_name, clf in tqdm(classifiers.items(), total=len(classifiers), desc=f'Classification: {combo_name}'):
            
            best_perf = {'acc_mean':0, 'acc_std':0, 'bacc_mean':0, 'bacc_std':0, 'spec_mean':0, 'spec_std':0, 
                         'sens_mean':0, 'sens_std':0, 'auc_mean':0, 'auc_std':0, 'features':None}
            
            for feat in list_feat:
                if len(feat) == 0:
                    continue
                
                accs, baccs, specs, senss, aucs = [], [], [], [], []
                for fold_idx in [1,2,3,4,5]:
                    X_train = data.loc[data['fold_idx']!=fold_idx, feat].values
                    y_train = data.loc[data['fold_idx']!=fold_idx, 'category'].values
                    X_test = data.loc[data['fold_idx']==fold_idx, feat].values
                    y_test = data.loc[data['fold_idx']==fold_idx, 'category'].values
                    
                    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
                    X_train = scaling.transform(X_train)
                    X_test = scaling.transform(X_test)
                    
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
                
                if auc_mean >= best_perf['auc_mean']:
                    best_perf = {
                        'acc_mean': acc_mean, 'acc_std': acc_std,
                        'bacc_mean': bacc_mean, 'bacc_std': bacc_std,
                        'spec_mean': spec_mean, 'spec_std': spec_std,
                        'sens_mean': sens_mean, 'sens_std': sens_std,
                        'auc_mean': auc_mean, 'auc_std': auc_std,
                        'features': feat,
                    }
                    
            for metric in ['acc', 'bacc', 'spec', 'sens', 'auc']:
                row[f'{classifier_name}_{metric}_mean'] = best_perf[f'{metric}_mean']
                row[f'{classifier_name}_{metric}_std'] = best_perf[f'{metric}_std']
                row_simple[f'{classifier_name}_{metric}'] = f"{best_perf[f'{metric}_mean']:.3f}±{best_perf[f'{metric}_std']:.3f}"
            row[f'{classifier_name}_best_features'] = str(best_perf['features'])

        row = pd.DataFrame(row)
        row_simple = pd.DataFrame(row_simple)
        results = pd.concat([results, row], axis=0)
        results_simple = pd.concat([results_simple, row_simple], axis=0)

    results.to_csv(results_csv, index=False)
    results_simple.to_excel(results_csv.replace('.csv', '_simple.xlsx'), index=False)


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
                    df_model[f'age_pred{col_suffix}_{fold_idx}_times_chronological_age'] = df_model[f'age_pred{col_suffix}_{fold_idx}'] * df_model['age_gt']
                    df_model[f'age_pred{col_suffix}_{fold_idx}_bag'] = df_model[f'age_pred{col_suffix}_{fold_idx}'] - df_model['age_gt']
                    df_model[f'age_pred{col_suffix}_{fold_idx}_bag_times_chronological_age'] = df_model[f'age_pred{col_suffix}_{fold_idx}_bag'] * df_model['age_gt']
                else:
                    tmp = pd.read_csv(pred_csv)
                    tmp = tmp.groupby(['dataset','subject','session','age_gt'])['age_pred'].mean().reset_index()
                    tmp = tmp.rename(columns={'age_pred': f'age_pred{col_suffix}_{fold_idx}'})
                    tmp[f'age_pred{col_suffix}_{fold_idx}_times_chronological_age'] = tmp[f'age_pred{col_suffix}_{fold_idx}'] * tmp['age_gt']
                    tmp[f'age_pred{col_suffix}_{fold_idx}_bag'] = tmp[f'age_pred{col_suffix}_{fold_idx}'] - tmp['age_gt']
                    tmp[f'age_pred{col_suffix}_{fold_idx}_bag_times_chronological_age'] = tmp[f'age_pred{col_suffix}_{fold_idx}_bag'] * tmp['age_gt']
                    df_model = df_model.merge(tmp, on=['dataset','subject','session','age_gt'])
            df_model[f'age_pred{col_suffix}_mean'] = df_model[[f'age_pred{col_suffix}_{fold_idx}' for fold_idx in folds]].mean(axis=1)
            df_model[f'age_pred{col_suffix}_mean_times_chronological_age'] = df_model[f'age_pred{col_suffix}_mean'] * df_model['age_gt']
            df_model[f'age_pred{col_suffix}_mean_bag'] = df_model[f'age_pred{col_suffix}_mean'] - df_model['age_gt']
            df_model[f'age_pred{col_suffix}_mean_bag_times_chronological_age'] = df_model[f'age_pred{col_suffix}_mean_bag'] * df_model['age_gt']
            
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
        
        print("Age distribution of each category:")
        for c in df['category'].unique():    
            print(f"--------------{c}--------------")
            print(df.loc[df['category']==c, 'age_gt'].describe())
            
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
# df.to_csv('experiments/2024-03-06_Cross_Sectional_CN_MCI_Dementia_Classification/data_v3.csv', index=False)
df = pd.read_csv('experiments/2024-03-06_Cross_Sectional_CN_MCI_Dementia_Classification/data_v3.csv')

# feature combinations
feat_elements = {}
for model in dict_models.keys():
    col_suffix = dict_models[model]['col_suffix']
    feat_elements[model] = {
        'prediction': [
            [],
            [f'age_pred{col_suffix}_mean'],
            [f'age_pred{col_suffix}_{i}' for i in [1,2,3,4,5]],
            [f'age_pred{col_suffix}_mean'] + [f'age_pred{col_suffix}_{i}' for i in [1,2,3,4,5]],
        ],
        'prediction_interaction': [
            [],
            [f'age_pred{col_suffix}_mean_times_chronological_age'],
            [f'age_pred{col_suffix}_{i}_times_chronological_age' for i in [1,2,3,4,5]],
            [f'age_pred{col_suffix}_mean_times_chronological_age'] + [f'age_pred{col_suffix}_{i}_times_chronological_age' for i in [1,2,3,4,5]],
        ],
        'bag': [
            [],
            [f'age_pred{col_suffix}_mean_bag'],
            [f'age_pred{col_suffix}_{i}_bag' for i in [1,2,3,4,5]],
            [f'age_pred{col_suffix}_mean_bag'] + [f'age_pred{col_suffix}_{i}_bag' for i in [1,2,3,4,5]],
        ],
        'bag_interaction': [
            [],
            [f'age_pred{col_suffix}_mean_bag_times_chronological_age'],
            [f'age_pred{col_suffix}_{i}_bag_times_chronological_age' for i in [1,2,3,4,5]],
            [f'age_pred{col_suffix}_mean_bag_times_chronological_age'] + [f'age_pred{col_suffix}_{i}_bag_times_chronological_age' for i in [1,2,3,4,5]],
        ],
    }

feat_combo = {}
feat_combo['chrono age only'] = [['age_gt']]
feat_combo['WM age only'] = feat_elements['WM age model']['prediction']
feat_combo['WM age (contaminated) only'] = feat_elements['WM age model (contaminated with GM age features)']['prediction']
feat_combo['GM age only'] = feat_elements['GM age model (ours)']['prediction']
feat_combo['GM age (TSAN) only'] = feat_elements['GM age model (TSAN)']['prediction']
feat_combo['chrono + WM age'] = []
for v in ['prediction', 'bag']:
    for i in feat_elements['WM age model'][v]:
        for j in feat_elements['WM age model'][f'{v}_interaction']:
            if i + j == []:
                continue  # there must be at least one feature from WM age model
            feat_combo['chrono + WM age'].append(['age_gt'] + i + j)
feat_combo['chrono + GM age'] = []
for v in ['prediction', 'bag']:
    for i in feat_elements['GM age model (ours)'][v]:
        for j in feat_elements['GM age model (ours)'][f'{v}_interaction']:
            if i + j == []:
                continue  # there must be at least one feature from GM age model
            feat_combo['chrono + GM age'].append(['age_gt'] + i + j)
feat_combo['chrono + WM age + GM age'] = []
for v in ['prediction', 'bag']:
    for i in feat_elements['WM age model'][f'{v}_interaction']:
        for j in feat_elements['WM age model'][v]:
            if i + j == []:
                continue  # there must be at least one feature from WM age model
            for k in feat_elements['GM age model (ours)'][f'{v}_interaction']:
                for l in feat_elements['GM age model (ours)'][v]:
                    if k + l == []:
                        continue  # there must be at least one feature from GM age model
                    feat_combo['chrono + WM age + GM age'].append(['age_gt'] + i + j + k + l)
                    
# classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Linear SVM': SVC(kernel="linear", C=1, probability=True, random_state=42),
    'RBF SVM': SVC(kernel='rbf', C=1, probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(
        max_depth=5, n_estimators=10, random_state=42
    ),
    'Naive Bayes': GaussianNB(),
    'Nearest Neighbors': KNeighborsClassifier(3),
}

# Task 1: CN vs AD
data = df.loc[df['category'].isin(['CN', 'AD']), ].copy()
data['category'] = data['category'].map({'CN': 0, 'AD': 1})
run_classification_experiments(
    data=data, feat_combo=feat_combo, classifiers=classifiers, 
    results_csv='experiments/2024-03-06_Cross_Sectional_CN_MCI_Dementia_Classification/results_CN_vs_AD_v3.csv')

# Task 2: CN vs MCI
data = df.loc[df['category'].isin(['CN', 'MCI']), ].copy()
data['category'] = data['category'].map({'CN': 0, 'MCI': 1})
run_classification_experiments(
    data=data, feat_combo=feat_combo, classifiers=classifiers, 
    results_csv='experiments/2024-03-06_Cross_Sectional_CN_MCI_Dementia_Classification/results_CN_vs_MCI_v3.csv')

# Task 3: CN vs CN*
data = df.loc[df['category'].isin(['CN', 'CN*']), ].copy()
data['category'] = data['category'].map({'CN': 0, 'CN*': 1})
run_classification_experiments(
    data=data, feat_combo=feat_combo, classifiers=classifiers, 
    results_csv='experiments/2024-03-06_Cross_Sectional_CN_MCI_Dementia_Classification/results_CN_vs_CNstar_v3.csv')