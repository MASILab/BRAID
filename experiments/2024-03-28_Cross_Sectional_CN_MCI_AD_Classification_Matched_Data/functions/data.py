import re
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

class DataPreparation:
    def __init__(self, dict_models, databank_csv):
        self.dict_models = dict_models
        self.databank = pd.read_csv(databank_csv)
        
    def load_predictions_of_all_models(self):
        """ Load dataframes from each fold of each model and combine them into a 
        single wide dataframe, with diagnosis information collected from the databank.
        """
        for i, model in enumerate(self.dict_models.keys()):
            pred_root = Path(self.dict_models[model]['prediction_root'])
            col_suffix = self.dict_models[model]['col_suffix']
            
            for fold_idx in tqdm([1,2,3,4,5], desc=f'Loading data for {model}'):
                pred_csv = pred_root / f"predicted_age_fold-{fold_idx}_test_bc.csv"
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
        return df
    
    def assign_fine_category_label(self, df):
        """ The diagnosis label is a noisy label. Now we assign a new label, "category", based on some rules.
        We create two columns, "category_criteria_1" and "category_criteria_2", that differ in the rule of assigning the CN category.
        We hope that the new label can be more precise. We will have the following four categories:
        
        - CN: there is only "cognitively normal (CN)" in the diagnosis history of the subject. (category_criteria_2 = 'CN')
            Note that we could apply a more strict rule, which further requires that the subject has at least two sessions. 
            In that case, we can discard the last session, at which we are not sure if the subject will stay CN in the future,
            and use the remaining "certain" sessions. (category_criteria_1 = 'CN')
            We used "category_criteria_1" in the matched cohort experiment, where the bottleneck of data is the CN* category, 
            so losing some CN subjects would not affect much. However, in this experiment, we might want to use "category_criteria_2"
            in "CN vs. AD" and especially in "CN vs. MCI" classification so we can have more samples after matching.
        - CN*: CN in the current session, but is diagnosed with MCI/AD in the next session which is XX ± XX years later.
        - MCI: mild cognitive impairment
        - AD: Alzheimer's disease
        """
        df['subj'] = df['dataset'] + '_' + df['subject']
        df['category_criteria_1'] = None
        df['category_criteria_2'] = None
        filter_age = df['age'].between(45, 90)

        # CN*: the minority category goes first
        for subj in df.loc[filter_age, 'subj'].unique():
            rows_subj = df.loc[df['subj']==subj, ].copy()
            rows_subj = rows_subj.sort_values(by='age')
            for i in range(len(rows_subj.index)-1):
                if (rows_subj.iloc[i]['diagnosis']=='normal') & (rows_subj.iloc[i+1]['diagnosis'] in ['MCI', 'dementia']):
                    df.loc[(df['subj']==subj) & (df['age']==rows_subj.iloc[i]['age']), ['category_criteria_1','category_criteria_2']] = 'CN*'
        # CN
        for subj in df.loc[filter_age & (df['diagnosis']=='normal'), 'subj'].unique():
            if len(df.loc[df['subj']==subj, 'diagnosis'].unique())==1:  # there is only 'normal' in diagnosis history
                df.loc[df['subj']==subj, 'category_criteria_2'] = 'CN'
                if len(df.loc[df['subj']==subj, 'age'].unique())>=2:  # at least two sessions are available
                    df.loc[(df['subj']==subj) & (df['age']!=df.loc[df['subj']==subj,'age'].max()), 'category_criteria_1'] = 'CN'  # pick all but the last session (which is uncertain if it progresses to MCI/AD)
        # MCI
        for subj in df.loc[filter_age & (df['diagnosis']=='MCI'), 'subj'].unique():
            if subj in df.loc[df['category_criteria_1']=='CN*', 'subj'].unique():
                continue   # CN* is the minority category, we don't want to overwrite the subjects in CN* to MCI
            df.loc[(df['subj']==subj)&(df['diagnosis']=='MCI'), ['category_criteria_1', 'category_criteria_2']] = 'MCI'
        # AD
        for subj in df.loc[filter_age & (df['diagnosis']=='dementia'), 'subj'].unique():
            if subj in df.loc[df['category_criteria_1']=='CN*', 'subj'].unique():
                continue
            df.loc[(df['subj']==subj)&(df['diagnosis']=='dementia'), ['category_criteria_1', 'category_criteria_2']] = 'AD'
        
        print("Category assignment done.")
        for i in [1,2]:
            print(f"Summary of category_criteria_{i}:\n{df['category_criteria_'+str(i)].value_counts()}")
            
        return df

    def feature_engineering(self, df):
        """ Create new features from current data. Convert categorical data to binary.
        """
        # Convert sex to binary
        df['sex'] = df['sex'].map({'female': 0, 'male': 1})

        # Mean value of age predictions from all folds for each model
        for model in self.dict_models.keys():
            col_suffix = self.dict_models[model]['col_suffix']
            df[f'age_pred{col_suffix}_mean'] = df[[f'age_pred{col_suffix}_{fold_idx}' for fold_idx in [1,2,3,4,5]]].mean(axis=1)
        
        # Brain age gap (BAG)
        for model in self.dict_models.keys():
            col_suffix = self.dict_models[model]['col_suffix']
            # each fold
            for fold_idx in [1,2,3,4,5]:
                df[f'age_pred{col_suffix}_{fold_idx}_bag'] = df[f'age_pred{col_suffix}_{fold_idx}'] - df['age']
            # mean of all folds
            df[f'age_pred{col_suffix}_mean_bag'] = df[f'age_pred{col_suffix}_mean'] - df['age']
        
        # BAG change rate_i+1 = (BAG_i+1 - BAG_i) / (age_i+1 - age_i)
        for model in self.dict_models.keys():
            col_suffix = self.dict_models[model]['col_suffix']
            for fold_idx in [1,2,3,4,5]:
                df[f'age_pred{col_suffix}_{fold_idx}_bag_change_rate'] = None
            df[f'age_pred{col_suffix}_mean_bag_change_rate'] = None
        
        for subj in df['subj'].unique():
            rows_subj = df.loc[df['subj']==subj, ].copy()
            rows_subj = rows_subj.sort_values(by='age')
            for i in range(len(rows_subj.index)-1):
                for model in self.dict_models.keys():
                    col_suffix = self.dict_models[model]['col_suffix']
                    for fold_idx in [1,2,3,4,5]:
                        delta_bag = rows_subj.iloc[i+1][f'age_pred{col_suffix}_{fold_idx}_bag'] - rows_subj.iloc[i][f'age_pred{col_suffix}_{fold_idx}_bag']
                        interval = rows_subj.iloc[i+1]['age'] - rows_subj.iloc[i]['age']
                        df.loc[(df['subj']==subj)&(df['age']==rows_subj.iloc[i+1]['age']), f'age_pred{col_suffix}_{fold_idx}_bag_change_rate'] = (delta_bag / interval) if interval > 0 else None
                    # mean of all folds
                    delta_bag = rows_subj.iloc[i+1][f'age_pred{col_suffix}_mean_bag'] - rows_subj.iloc[i][f'age_pred{col_suffix}_mean_bag']
                    interval = rows_subj.iloc[i+1]['age'] - rows_subj.iloc[i]['age']
                    df.loc[(df['subj']==subj)&(df['age']==rows_subj.iloc[i+1]['age']), f'age_pred{col_suffix}_mean_bag_change_rate'] = (delta_bag / interval) if interval > 0 else None
        
        # (chronological age*BAG) and (chronological age*BAG change rate)
        for model in self.dict_models.keys():
            col_suffix = self.dict_models[model]['col_suffix']
            for fold_idx in [1,2,3,4,5]:
                df[f'age_pred{col_suffix}_{fold_idx}_bag_multiply_age'] = df[f'age_pred{col_suffix}_{fold_idx}_bag'] * df['age']
                df[f'age_pred{col_suffix}_{fold_idx}_bag_change_rate_multiply_age'] = df[f'age_pred{col_suffix}_{fold_idx}_bag_change_rate'] * df['age']
            df[f'age_pred{col_suffix}_mean_bag_multiply_age'] = df[f'age_pred{col_suffix}_mean_bag'] * df['age']
            df[f'age_pred{col_suffix}_mean_bag_change_rate_multiply_age'] = df[f'age_pred{col_suffix}_mean_bag_change_rate'] * df['age']
            
        return df
    
    def draw_histograms_columns(self, df, cols_exclude=['dataset','subject','session'], 
                                save_dir='experiments/2024-03-28_Cross_Sectional_CN_MCI_AD_Classification_Matched_Data/figs/histograms'):
        """ Draw histograms for all columns in the dataframe.
        """
        
        cols_plot = [col for col in df.columns if col not in cols_exclude]
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for col in tqdm(cols_plot, desc='Drawing histograms'):
            fig, ax = plt.subplots()
            sns.histplot(data=df, x=col, ax=ax)
            fig.savefig(save_dir / f"{col}.png")
            plt.close('all')
            
    # def match_data(self, category_col, match_order, age_diff_threshold=1):
    #     pass




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
df = d.load_predictions_of_all_models()
df = d.retrieve_diagnosis_label(df)
df = d.assign_fine_category_label(df)
df = d.feature_engineering(df)
df.to_csv('tmp.csv', index=False)
d.draw_histograms_columns(df)