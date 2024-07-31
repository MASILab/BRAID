import pdb
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from warnings import simplefilter
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
                df_model[f'age_pred{col_suffix}'] = df_model[[f'age_pred{col_suffix}_{fold_idx}' for fold_idx in [1,2,3,4,5]]].mean(axis=1)
            df_model = df_model[['dataset','subject','session','sex','age',f'age_pred{col_suffix}']]
            print(f'Loaded data for {model}, shape: {df_model.shape}')
            
            if i == 0:
                df = df_model.copy()
            else:
                df = df.merge(df_model.copy(), on=['dataset','subject','session','sex','age'])
        
        # remove duplicated rows
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
        """ Create new features from current data.
        """
        # Brain age gap (BAG)
        for model in self.dict_models.keys():
            col_suffix = self.dict_models[model]['col_suffix']
            df[f'age_pred{col_suffix}_bag'] = df[f'age_pred{col_suffix}'] - df['age']
        
        # BAG change rate_i = (BAG_i+1 - BAG_i) / (age_i+1 - age_i)
        for model in self.dict_models.keys():
            col_suffix = self.dict_models[model]['col_suffix']
            df[f'age_pred{col_suffix}_bag_change_rate'] = None
        
        for subj in df['subj'].unique():
            rows_subj = df.loc[df['subj']==subj, ].copy()
            rows_subj = rows_subj.sort_values(by='age')
            for i in range(len(rows_subj.index)-1):
                interval = rows_subj.iloc[i+1]['age'] - rows_subj.iloc[i]['age']  # age_i+1 - age_i
                
                for model in self.dict_models.keys():
                    col_suffix = self.dict_models[model]['col_suffix']
                    delta_bag = rows_subj.iloc[i+1][f'age_pred{col_suffix}_bag'] - rows_subj.iloc[i][f'age_pred{col_suffix}_bag']
                    df.loc[(df['subj']==subj)&(df['age']==rows_subj.iloc[i]['age']), f'age_pred{col_suffix}_bag_change_rate'] = (delta_bag / interval) if interval > 0 else None
        
        # Impute NaN values with the value from the closest session (if there is any) of the subject
        cols = [col for col in df.columns if '_bag_change_rate' in col]
        for subj in df['subj'].unique():
            ages = sorted(df.loc[df['subj']==subj, 'age'].unique())
            if len(ages) == 1:
                continue
            for col in cols:
                df.loc[(df['subj']==subj)&(df['age']==ages[-1]), col] = df.loc[(df['subj']==subj)&(df['age']==ages[-2]), col].values[0]
        
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
    
    def assign_category_label(self, df):
        # Assign category label: CN, CN*, MCI, AD
        df['category'] = None
        df.loc[df['cn_label']>=0.5, 'category'] = 'CN'
        df.loc[(df['time_to_MCI']>0)&(df['time_to_MCI']<=10), 'category'] = 'CN*'  # (0,6)
        for disease in ['AD', 'MCI']:
            subj_assigned = df.loc[df['category'].notna(), 'subj'].unique()  # do not reuse the subjects of the minority group
            df.loc[(df['diagnosis']==disease)&(~df['subj'].isin(subj_assigned)), 'category'] = disease
        
        return df


def get_matched_cohort(df, age_diff_threshold=1):    
    # Greedy matching
    search_order = ['CN*', 'AD', 'MCI', 'CN']
    dfs_pool = {}
    dfs_pool['CN'] = df.loc[df['category']=='CN', ].sort_values(by=['subj', 'time_to_last_cn'], ascending=True).copy()
    dfs_pool['CN*'] = df.loc[df['category']=='CN*', ].sort_values(by=['subj','time_to_MCI'], ascending=True).copy()
    dfs_pool['MCI'] = df.loc[df['category']=='MCI', ].sort_values(by=['subj', 'age'], ascending=True).copy()
    dfs_pool['AD'] = df.loc[df['category']=='AD', ].sort_values(by=['subj', 'age'], ascending=True).copy()
    dfs_matched = {c: pd.DataFrame() for c in search_order}
    match_id = 0

    for _, row in tqdm(dfs_pool[search_order[0]].iterrows(), total=len(dfs_pool[search_order[0]].index), desc='Greedy matching'):
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
                if (c=='CN') and (row_c['time_to_last_cn'] < row['time_to_MCI'] - age_diff_threshold):
                    continue
                
                age_diff = abs(row['age'] - row_c['age'])
                if age_diff < smallest_age_diff:
                    smallest_age_diff = age_diff
                    tmp_matched[c] = row_c
            if tmp_matched[c] is not None:
                used_subjs.append(tmp_matched[c]['subj'])
        
        if all([tmp_matched[c] is not None for c in search_order[1:]]):  # this sample has been matched in all categories
            for c in search_order:
                r = row.to_frame().T if c==search_order[0] else tmp_matched[c].to_frame().T
                r['match_id'] = match_id
                dfs_matched[c] = pd.concat([dfs_matched[c], r])                
            match_id += 1
    print(f'--------> Greedy matching completed. Found {match_id} pairs.')
    
    # sanity check
    for i, c in enumerate(search_order):
        for j in range(i, len(search_order)):
            if i == j:
                # make sure that no repeated subjects are in the same category
                assert len(dfs_matched[c]['subj'].unique()) == len(dfs_matched[c].index), f"Repeated subjects in {c}"
            else:
                # no overlapping subjects
                assert len(set(dfs_matched[c]['subj'].unique()).intersection(set(dfs_matched[search_order[j]]['subj'].unique()))) == 0, f"Overlapping subjects between {c} and {search_order[j]}"
                
    # merge into one dataframe
    df_matched = pd.DataFrame()
    for c in search_order:
        df_matched = pd.concat([df_matched, dfs_matched[c]], ignore_index=True)
    df_matched = df_matched.sort_values(by=['match_id', 'category'], ignore_index=True)
    
    return df_matched


def make_quiver_plot(df, x_col_name, y_col_name, u_col_name, v_col_name, png):

    color_dict = {
        'CN': 'tab:green',
        'CN*': 'gold',
        'MCI': 'tab:orange',
        'AD': 'tab:red',
    }
    df = df.loc[df['category'].notna(), ].copy()
    colors = df['category'].map(color_dict)

    dict_axis_labels = {
        'age_pred_wm_age_nonlinear_bag': 'WM age nonrigid\nbrain age gap (years)',
        'age_pred_wm_age_affine_bag': 'WM age affine\nbrain age gap (years)',
        'age_pred_gm_age_ours_bag': 'GM age (ours)\nbrain age gap (years)',
        'age_pred_gm_age_tsan_bag': 'GM age (TSAN)\nbrain age gap (years)',
        'age_pred_gm_age_dbn_bag': 'GM age (DBN)\nbrain age gap (years)',
    }

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.quiver(
        df[x_col_name].to_numpy(dtype='float16'), df[y_col_name].to_numpy(dtype='float16'), 
        df[u_col_name].to_numpy(dtype='float16'), df[v_col_name].to_numpy(dtype='float16'), 
        color=colors, angles='xy', scale=1, scale_units='xy', units='xy'
        )
    
    ax.set_aspect('equal')

    # Add legend
    for category, color in color_dict.items():
        ax.scatter([], [], c=color, label=category)
    ax.legend(title='Category')

    # Set labels and title
    ax.set_xlabel(dict_axis_labels[x_col_name], fontsize=9, fontname='DejaVu Sans')
    ax.set_ylabel(dict_axis_labels[y_col_name], fontsize=9, fontname='DejaVu Sans')
    fig.savefig(png, dpi=300)
    
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for category, color in color_dict.items():
        ax.scatter(
            x = df.loc[df['category']==category, x_col_name].to_numpy(dtype='float16'), 
            y = df.loc[df['category']==category, y_col_name].to_numpy(dtype='float16'), 
            s = 2,
            c = color,
            label=category,
            alpha=0.75)    
    ax.legend(title='Category')
    ax.set_aspect('equal')
    ax.set_xlabel(dict_axis_labels[x_col_name], fontsize=9, fontname='DejaVu Sans')
    ax.set_ylabel(dict_axis_labels[y_col_name], fontsize=9, fontname='DejaVu Sans')
    fig.savefig(png.replace('.png', '_scatter.png'), dpi=300)
    plt.close('all')


if __name__ == '__main__':
    # Load data
    output_fn = 'reports/figures/2024-07-30_Quiver_plot_wm_gm_bag/data/data_prep.csv'
    if Path(output_fn).is_file():
        df = pd.read_csv(output_fn)
    else:
        data_prep = DataPreparation(roster_brain_age_models(), '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv')
        df = data_prep.load_predictions_of_all_models()
        df = data_prep.retrieve_diagnosis_label(df)
        df = data_prep.assign_cn_label(df)
        df = data_prep.feature_engineering(df)
        df = data_prep.mark_progression_subjects_out(df)
        df = data_prep.assign_category_label(df)
        df.to_csv(output_fn, index=False)
    
    # filter out NaN and outliers
    for x, y in [('age_pred_gm_age_ours_bag', 'age_pred_wm_age_nonlinear_bag'),
                 ('age_pred_gm_age_ours_bag', 'age_pred_wm_age_affine_bag'),
                 ('age_pred_gm_age_dbn_bag', 'age_pred_wm_age_nonlinear_bag'),
                 ('age_pred_gm_age_dbn_bag', 'age_pred_wm_age_affine_bag')]:
        
        data = df.copy()
        x_cr, y_cr = x.replace('_bag', '_bag_change_rate'), y.replace('_bag', '_bag_change_rate')
        
        for col in [x_cr, y_cr]:
            for cat in data['category'].unique():
                q1 = data.loc[data['category']==cat, col].quantile(0.25)
                q3 = data.loc[data['category']==cat, col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outlier_indices = data.loc[(data['category']==cat) & ((data[col] < lower_bound) | (data[col] > upper_bound))].index
                data = data.drop(outlier_indices)
                
        for age_range in [[45, 90], [45, 75], [50, 80], [55, 85], [60, 90], [55, 75], [65, 85], [75, 90], [45, 60], [55, 70], [65, 80], [75, 90]]:
            data_filtered = data.loc[data['age'].between(age_range[0], age_range[1]), ].copy()
            try:
                data_matched = get_matched_cohort(data_filtered, age_diff_threshold=1)
            except:
                continue
            
            make_quiver_plot(data_matched, x, y, x_cr, y_cr, f'reports/figures/2024-07-30_Quiver_plot_wm_gm_bag/figs/{age_range[0]}-{age_range[1]}_{x}_{y}.png')