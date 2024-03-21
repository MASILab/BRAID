import pdb
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

class WMGMAgeLoader:
    def __init__(self, wm_pred_root, gm_pred_root, fn_pattern, databank_dti_csv):
        self.wm_pred_root = wm_pred_root
        self.gm_pred_root = gm_pred_root
        self.fn_pattern = fn_pattern
        self.databank_dti = pd.read_csv(databank_dti_csv)
    
    def find_pred_csvs(self, pred_root, fn_pattern):
        csvs = sorted(Path(pred_root).glob(fn_pattern))
        return csvs
    
    def merge_predictions(self, csvs):
        """ Average predictions across scans of the same session.
        Merge predictions from all five folds (use mean value).
        """
        for i, csv in enumerate(csvs):
            match = re.search(r'fold-(\d+)', csv.name)
            fold_idx = int(match.group(1))
            assert fold_idx == i+1, f"Mismatch between fold index and csv index: {fold_idx} vs {i+1}"
            
            if fold_idx == 1:
                df = pd.read_csv(csv)
                df = df.groupby(['dataset','subject','session','sex','age'])['age_pred'].mean().reset_index()
                df = df.rename(columns={'age_pred': f'age_pred_{fold_idx}'})
            else:
                tmp = pd.read_csv(csv)
                tmp = tmp.groupby(['dataset','subject','session','sex','age'])['age_pred'].mean().reset_index()
                tmp = tmp.rename(columns={'age_pred': f'age_pred_{fold_idx}'})
                df = df.merge(tmp, on=['dataset','subject','session','sex','age'])
        df['age_pred_mean'] = df[[f'age_pred_{i+1}' for i in range(len(csvs))]].mean(axis=1)
        
        return df[['dataset','subject','session','sex','age','age_pred_mean']].copy()
    
    def collect_diagnosis(self, df, databank):
        df['diagnosis'] = None
        for i, row in tqdm(df.iterrows(), total=len(df.index), desc='Retrieve diagnosis information'):
            loc_filter = (databank['dataset']==row['dataset']) & (databank['subject']==row['subject']) & ((databank['session']==row['session']) | (databank['session'].isnull()))
            if row['dataset'] in ['UKBB']:
                control_label = databank.loc[loc_filter, 'control_label'].values[0]
                df.loc[i, 'diagnosis'] = 'normal' if control_label == 1 else None
            else:
                df.loc[i, 'diagnosis'] = databank.loc[loc_filter, 'diagnosis_simple'].values[0]
        return df

    def collect_both_wmgm_and_assign_category(self, random_seed=42):
        wm_pred_csvs = self.find_pred_csvs(self.wm_pred_root, self.fn_pattern)
        gm_pred_csvs = self.find_pred_csvs(self.gm_pred_root, self.fn_pattern)
        
        df_wm = self.merge_predictions(wm_pred_csvs)
        df_gm = self.merge_predictions(gm_pred_csvs)
        df = df_wm.merge(df_gm[['dataset','subject','session','age_pred_mean']], on=['dataset','subject','session'], suffixes=('_wm', '_gm'))
        df = self.collect_diagnosis(df, self.databank_dti)
        df['wm_gm_diff'] = df['age_pred_mean_wm'] - df['age_pred_mean_gm']
        df['wm_gm_mean'] = (df['age_pred_mean_wm'] + df['age_pred_mean_gm']) / 2
        
        # assign categories: CN*, CN, MCI, AD, there should be no overlapping subjects between categories
        df['subj'] = df['dataset'] + '_' + df['subject']
        df['category_criteria_1'] = None  # more strict criteria on CN category
        df['category_criteria_2'] = None  # backup plan for more samples after greedy matching
        filter_age = df['age'].between(45, 90)

        # CN*: CN in the current session, but is diagnosed with MCI/AD in the next session XX ± XX years later
        for subj in df.loc[filter_age, 'subj'].unique():
            rows_subj = df.loc[df['subj']==subj, ].copy()
            rows_subj = rows_subj.sort_values(by='age')
            for i in range(len(rows_subj.index)-1):
                if (rows_subj.iloc[i]['diagnosis']=='normal') & (rows_subj.iloc[i+1]['diagnosis'] in ['MCI', 'dementia']):
                    df.loc[(df['subj']==subj) & (df['age']==rows_subj.iloc[i]['age']), ['category_criteria_1', 'category_criteria_2']] = 'CN*'
        # CN
        for subj in df.loc[filter_age & (df['diagnosis']=='normal'), 'subj'].unique():
            if len(df.loc[df['subj']==subj, 'diagnosis'].unique())==1:  # there is only 'normal' in diagnosis history
                df.loc[df['subj']==subj, 'category_criteria_2'] = 'CN'
                if len(df.loc[df['subj']==subj, 'age'].unique())>=2:  # at least two sessions are available
                    df.loc[(df['subj']==subj) & (df['age']!=df.loc[df['subj']==subj,'age'].max()), 'category_criteria_1'] = 'CN'  # pick all but the last session (which is uncertain if it progresses to MCI/AD)
        # MCI
        for subj in df.loc[filter_age & (df['diagnosis']=='MCI'), 'subj'].unique():
            if subj in df.loc[df['category_criteria_1']=='CN*', 'subj'].unique():
                continue   # CN* is the minority group, we don't want to use the subjects in CN* for MCI
            df.loc[(df['subj']==subj)&(df['diagnosis']=='MCI'), ['category_criteria_1', 'category_criteria_2']] = 'MCI'
            
        # AD
        for subj in df.loc[filter_age & (df['diagnosis']=='dementia'), 'subj'].unique():
            if subj in df.loc[df['category_criteria_1']=='CN*', 'subj'].unique():
                continue
            df.loc[(df['subj']==subj)&(df['diagnosis']=='dementia'), ['category_criteria_1', 'category_criteria_2']] = 'AD'
            
        df = df.loc[filter_age, ]
        
        return df


def match_cohort(df, category_col='category_criteria_1', search_order=['CN*', 'AD', 'MCI', 'CN'], age_diff_threshold=1):

    # candidate pool and selected samples
    dfs_pool = {c: df.loc[df[category_col]==c, ].copy() for c in search_order}
    dfs_matched = {c: pd.DataFrame() for c in search_order}
    
    for i, row in tqdm(dfs_pool[search_order[0]].iterrows(), total=len(dfs_pool[search_order[0]].index), desc=f'Matching cohorts with {search_order[0]}'):
        used_subjs = []
        for c in search_order:
            if len(dfs_matched[c].index) == 0:
                continue
            else:
                used_subjs += dfs_matched[c]['subj'].unique().tolist()

        if row['subj'] in used_subjs:
            continue

        tmp_matched = {c: None for c in search_order[1:]}
        for c in search_order[1:]:
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
        
        if all([tmp_matched[c] is not None for c in search_order[1:]]):  # this sample has been matched in all categories
            for c in search_order[1:]:
                dfs_matched[c] = pd.concat([dfs_matched[c], tmp_matched[c].to_frame().T])
            dfs_matched[search_order[0]] = pd.concat([dfs_matched[search_order[0]], row.to_frame().T])
    
    # sanity check
    for i, c in enumerate(search_order):
        for j in range(i, len(search_order)):
            if i == j:
                # make sure that no repeated subjects are in the same category
                assert len(dfs_matched[c]['subj'].unique()) == len(dfs_matched[c].index), f"Repeated subjects in {c}"
            else:
                # no overlapping subjects
                assert len(set(dfs_matched[c]['subj'].unique()).intersection(set(dfs_matched[search_order[j]]['subj'].unique()))) == 0, f"Overlapping subjects between {c} and {search_order[j]}"            
    
    # report matching MAE
    ae = []
    for i in range(len(dfs_matched[search_order[0]].index)):
        ae.append(sum([abs(dfs_matched[c].iloc[i]['age']-dfs_matched[search_order[0]].iloc[i]['age']) for c in search_order[1:]]) / (len(search_order)-1))
    mae = sum(ae) / len(ae)
    print(f"Mean absolute error: {mae:.2f} years")
    
    # merge into one dataframe
    df_merge = pd.DataFrame()
    for c in search_order:
        df_merge = pd.concat([df_merge, dfs_matched[c]], ignore_index=True)

    return df_merge


b = WMGMAgeLoader(
    wm_pred_root='models/2024-02-07_ResNet101_BRAID_warp/predictions', 
    gm_pred_root='models/2024-02-07_T1wAge_ResNet101/predictions', 
    fn_pattern='predicted_age_fold-*_test_bc.csv', 
    databank_dti_csv='/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv',
)
df = b.collect_both_wmgm_and_assign_category()
df_matched = match_cohort(df, category_col='category_criteria_1', search_order=['CN*', 'AD', 'MCI', 'CN'], age_diff_threshold=1)
df_matched.to_csv('experiments/2024-03-20_Matched_Cohort_Linear_Model/data_matched_cohort.csv', index=False)
