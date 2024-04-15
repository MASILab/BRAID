import pdb
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

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
            - "time_since_AD": the time interval (in years) between the first AD diagnosis and the current time point, 
                        negative means before the diagnosis and vice versa.
            - "age_MCI": the age when the subject was diagnosed with MCI for the first time.
            - "time_since_MCI": the time interval (in years) between the first MCI diagnosis and the current time point, 
                        negative means before the diagnosis and vice versa.
        Note: subjects, whose diagnosis of available sessions begins with MCI or AD, are excluded.
        """
        
        df = df.loc[df['diagnosis'].isin(['normal', 'MCI', 'AD']), ].copy()
    
        if 'subj' not in df.columns:
            df['subj'] = df['dataset'] + '_' + df['subject']
        
        for disease in ['AD', 'MCI']:
            df[f'age_{disease}'] = None
            
            for subj in df.loc[df['diagnosis']==disease, 'subj'].unique():
                rows_subj = df.loc[df['subj']==subj, ].copy()
                rows_subj = rows_subj.sort_values(by='age')
                if rows_subj.iloc[0]['diagnosis'] != 'normal':
                    continue
                df.loc[df['subj']==subj, f'age_{disease}'] = rows_subj.loc[rows_subj['diagnosis']==disease, 'age'].min()
            df[f'time_since_{disease}'] = df['age'] - df[f'age_{disease}']
            
            num_subj = len(df.loc[df[f'age_{disease}'].notna(), 'subj'].unique())
            print(f'Found {num_subj} subjects with {disease} progression.')
        
        return df

    def visualize_data_points(self, df, png, disease='MCI'):
        """ Visualize the chronological ages (or intervals before AD/MCI) of the data points 
        of subjects who have progressed from cognitively normal to MCI or AD.
        """

        assert f"time_since_{disease}" in df.columns, f"Column 'time_since_{disease}' is not available."
        
        df = df.loc[df[f'age_{disease}'].notna(), ].copy()
        if 'subj' not in df.columns:
            df['subj'] = df['dataset'] + '_' + df['subject']
            
        df = df.sort_values(by='age')
        df['y_subject'] = None
        for i, subj in enumerate(df['subj'].unique()):
            df.loc[df['subj']==subj, 'y_subject'] = i
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 12))
        
        for ax_id, x_axis in enumerate(['age', f'time_since_{disease}']):            
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
            axes[ax_id].set_xlabel(f'{x_axis} (years)', fontsize=16, fontfamily='DejaVu Sans')
            axes[ax_id].set_ylabel('Subject', fontsize=16, fontfamily='DejaVu Sans')
        fig.savefig(png, dpi=300)
    
    def get_preclinical_subsets(self, df, disease='MCI', method='index cut', num_subsets=11):
        """ Sample num_subsets subsets of data points from the dataframe. Each subset represents 
        a collection of cross-sectional data points that are years before the first diagnosis of desease.
        """
        
        if 'subj' not in df.columns:
            df['subj'] = df['dataset'] + '_' + df['subject']
            
        dict_subsets = {i: pd.DataFrame() for i in range(num_subsets)}
        
        if method == 'index cut':
            for i in range(num_subsets):
                for subj in df.loc[df[f'time_since_{disease}'].notna(), 'subj'].unique():
                    rows_subj = df.loc[(df['subj']==subj) & (df[f'time_since_{disease}']<=0), ].copy()
                    rows_subj = rows_subj.sort_values(by=f'time_since_{disease}', ascending=False)
                    
                    num_dp = len(rows_subj)
                    assert num_dp >= 2, f"The number of qualified data points for {subj} is less than 2."
                    
                    dict_subsets[i] = pd.concat([dict_subsets[i], rows_subj.iloc[round(i*(num_dp-1)/(num_subsets-1)), ].to_frame().T])
        else:
            raise ValueError(f"Method '{method}' is not supported.")
        
        print(f"Sampled {num_subsets-1} subsets of preclinical data points and 1 subset of clinical data points.")
        for k in dict_subsets.keys():
            print(f"Subset {k}: {dict_subsets[k][f'time_since_{disease}'].mean():.2f} +/- {dict_subsets[k][f'time_since_{disease}'].std():.2f} to {disease}")
        
        return dict_subsets
    
    def get_matched_cn_data(self, df_main, df_subset, age_diff_threshold=1):
        """ Use greedy algorithm to sample a subset of cognitively normal data points from the main dataframe df_main.
        The subset matches the data points in df_subset in terms of age and sex.
            The greedy algorithm will prioritize using the subjects that are cognitively normal under the strict definition.
        If the search does not find any data points that satisfy the age_diff_threshold, 
        the greedy algorithm will then use the subjects that are cognitively normal under the loose definition.
            The subset will be concatenated with df_subset to form a new dataframe.
        """        
        if 'subj' not in df_main.columns:
            df_main['subj'] = df_main['dataset'] + '_' + df_main['subject']
        if 'subj' not in df_subset.columns:
            df_subset['subj'] = df_subset['dataset'] + '_' + df_subset['subject']
        
        df_matched = pd.DataFrame()
        match_id = 0
        
        for _, row in df_subset.iterrows():
            used_subj = [] if len(df_matched.index)==0 else df_matched['subj'].unique().tolist()
            assert row['subj'] not in used_subj, f"{row['subj']} appeared more than once, this should not happen."
            
            best_match = None
            best_diff = age_diff_threshold
            
            for _, row_c in df_main.loc[df_main['cn_label']==1, ].iterrows():
                if (row_c['subj'] in used_subj) or (row_c['sex'] != row['sex']):
                    continue
                age_diff = abs(row_c['age'] - row['age'])
                if age_diff < best_diff:
                    best_match = row_c
                    best_diff = age_diff
            
            if best_match is None:
                for _, row_c in df_main.loc[df_main['cn_label']==0.5, ].iterrows():
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
                    df_matched = pd.concat([df_matched, r])
                match_id += 1
        
        return df_matched