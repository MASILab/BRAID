import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

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
        
        # interactions (chronological age/sex with BAG/BAG change rate)
        for model in self.dict_models.keys():
            col_suffix = self.dict_models[model]['col_suffix']
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
    
    def draw_histograms_columns(self, df, cols_exclude=['dataset','subject','session'], 
                                save_dir='experiments/2024-03-28_Cross_Sectional_CN_MCI_AD_Classification_Matched_Data/figs/histograms'):
        """ Draw histograms for all columns in the dataframe except for the ones in cols_exclude.
        """
        
        cols_plot = [col for col in df.columns if col not in cols_exclude]
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for col in tqdm(cols_plot, desc='Drawing histograms'):
            fig, ax = plt.subplots()
            sns.histplot(data=df, x=col, ax=ax)
            fig.savefig(save_dir / f"{col}.png")
            plt.close('all')

    def match_data(self, df, category_col, match_order, age_diff_threshold=1):
        """ Use greedy algorithm to match data samples from the majority category to the minority category.
        match_order: list of categories in the order of matching. The first category is the minority category.
        """
        # candidate pool and selected samples
        dfs_pool = {c: df.loc[df[category_col]==c, ].copy() for c in match_order}
        dfs_matched = {c: pd.DataFrame() for c in match_order}
        match_id = 0

        for _, row in tqdm(dfs_pool[match_order[0]].iterrows(), total=len(dfs_pool[match_order[0]].index), desc=f'Matching cohorts with {match_order[0]}'):
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
                
        # sanity check
        for i, c in enumerate(match_order):
            for j in range(i, len(match_order)):
                if i == j:
                    # make sure that no repeated subjects are in the same category
                    assert len(dfs_matched[c]['subj'].unique()) == len(dfs_matched[c].index), f"Repeated subjects in {c}"
                else:
                    # no overlapping subjects
                    assert len(set(dfs_matched[c]['subj'].unique()).intersection(set(dfs_matched[match_order[j]]['subj'].unique()))) == 0, f"Overlapping subjects between {c} and {match_order[j]}"
        
        # report matching MAE
        ae = []
        for i in range(len(dfs_matched[match_order[0]].index)):
            ae.append(sum([abs(dfs_matched[c].iloc[i]['age']-dfs_matched[match_order[0]].iloc[i]['age']) for c in match_order[1:]]) / (len(match_order)-1))
        mae = sum(ae) / len(ae)
        print(f"Mean absolute error: {mae:.2f} years")
        
        # merge into one dataframe
        df_merge = pd.DataFrame()
        for c in match_order:
            df_merge = pd.concat([df_merge, dfs_matched[c]], ignore_index=True)

        return df_merge
    
    def visualize_matched_data_histogram(self, df, category_col, save_png, xlim, xticks, ylim, yticks):
        categories = df[category_col].unique()
        assert len(categories) == 2, "Only support two classes for now."
        categories = [c for c in ['CN', 'CN*', 'MCI', 'AD'] if c in categories]  # reorder

        if (df['age'].max() >= xlim[1]) or (df['age'].min() <= xlim[0]):
            print(f"Warning: current FOV does not cover all data (min: {df['age'].min()}, Max: {df['age'].max()})!")

        # hyperparameters for plotting
        fontsize = 9
        fontfamily = 'DejaVu Sans'
        df['sex'] = df['sex'].map({0:'female', 1:'male'})  # convert back to string
        palette_hist = {'male': 'tab:blue', 'female': 'tab:red'}        
        fig = plt.figure(figsize=(3.5, 1.7))
        gs = gridspec.GridSpec(2, 2, wspace=0, hspace=0.7, height_ratios=[1, 0.05])
        
        for i, cat in enumerate(categories):
            data = df.loc[df[category_col]==cat, ]
            ax = fig.add_subplot(gs[0, i])
            sns.histplot(data=data, x='age', hue='sex', palette=palette_hist, alpha=1, binwidth=1, multiple='stack', ax=ax, legend=False)
            ax.set_title(f"{cat} (N={data.shape[0]})", fontsize=fontsize, fontfamily=fontfamily)
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            ax.set_xlabel('')
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, fontsize=fontsize, fontfamily=fontfamily)

            if i == 0:
                ax.set_ylabel('count', fontsize=fontsize, fontfamily=fontfamily)
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticks, fontsize=fontsize, fontfamily=fontfamily)
            else:
                ax.set_ylabel('')
                ax.set_yticks([])
                ax.set_yticklabels([])

        # xlabel and legend
        ax = fig.add_subplot(gs[1, :])
        ax.text(0.5, 0.9, 'chronological age (years)', fontsize=fontsize, fontfamily=fontfamily, ha='center', va='center', transform=ax.transAxes)
        patch1 = mpatches.Patch(edgecolor='black', facecolor=palette_hist['male'], label='Male')
        patch2 = mpatches.Patch(edgecolor='black', facecolor=palette_hist['female'], label='Female')
        ax.legend(handles=[patch1, patch2], loc='upper center', fontsize=fontsize, frameon=False, ncol=2, bbox_to_anchor=(0.5, 0.5))
        ax.axis('off')

        # Save figure
        fig.savefig(save_png, dpi=600, bbox_inches='tight')
    
    def split_data_into_k_folds(self, df, category_col, num_folds=5, fold_col='fold_idx', random_state=42):
        """ split the dataset at the subject level for cross-validation,
        save the fold information in a seperate column 'fold_idx'
        """
        assert fold_col not in df.columns, f'Column {fold_col} already exists in the dataframe'

        df[fold_col] = None

        for c in df[category_col].unique():
            subj_category = df.loc[df[category_col]==c, 'subj'].unique().tolist()
            random.seed(random_state)
            random.shuffle(subj_category)
            indices = [int(i*len(subj_category)/num_folds) for i in range(num_folds+1)]
            
            for i in range(num_folds):
                df.loc[(df[category_col]==c) & df['subj'].isin(subj_category[indices[i]:indices[i+1]]), fold_col] = i+1
        assert df[fold_col].notna().all(), f'Not all samples are assigned to a fold'
        
        return df

def roster_brain_age_models():
    dict_models = {
        'WM age model': {
            'prediction_root': 'models/2024-02-07_ResNet101_BRAID_warp/predictions',
            'col_suffix': '_wm_age_purified',
            },
        'GM age model (ours)': {
            'prediction_root': 'models/2024-02-07_T1wAge_ResNet101/predictions',
            'col_suffix': '_gm_age_ours'
            },
        'WM age model (contaminated with GM age features)': {
            'prediction_root': 'models/2023-12-22_ResNet101/predictions',
            'col_suffix': '_wm_age_contaminated'
            },
    }
    return dict_models

def roster_feature_combinations(df):
    feat_combo = {'basic (chronological age + sex)': ['age', 'sex']}
    feat_combo['basic + WM age (purified)'] = ['age', 'sex'] + [col for col in df.columns if '_wm_age_purified' in col]
    feat_combo['basic + GM age'] = ['age', 'sex'] + [col for col in df.columns if '_gm_age_ours' in col]
    feat_combo['basic + WM age (contaminated)'] = ['age', 'sex'] + [col for col in df.columns if '_wm_age_contaminated' in col]
    feat_combo['basic + WM age (purified) + GM age'] = ['age', 'sex'] + [col for col in df.columns if '_wm_age_purified' in col] + [col for col in df.columns if '_gm_age_ours' in col]
    
    return feat_combo