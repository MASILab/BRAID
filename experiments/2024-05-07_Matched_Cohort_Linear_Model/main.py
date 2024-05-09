import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from tqdm import tqdm
from pathlib import Path
from scipy.stats import wilcoxon

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
    
    
def get_matched_cohort(df, cnstar_threshold=(0,10), age_diff_threshold=1):
    # Assign category label: CN, CN*, MCI, AD
    df['category'] = None
    df.loc[df['cn_label']>=0.5, 'category'] = 'CN'
    df.loc[(df['time_to_MCI']>cnstar_threshold[0])&(df['time_to_MCI']<=cnstar_threshold[1]), 'category'] = 'CN*'
    for disease in ['AD', 'MCI']:
        subj_assigned = df.loc[df['category'].notna(), 'subj'].unique()  # do not reuse the subjects of the minority group
        df.loc[(df['diagnosis']==disease)&(~df['subj'].isin(subj_assigned)), 'category'] = disease
    
    # Filter out data points that fall out of the age range
    df = df.loc[df['age'].between(45, 90), ].copy()
    
    # Filter out outliers for each category
    df['wm_gm_diff'] = df['age_pred_wm_age_nonlinear'] - df['age_pred_gm_age_ours']
    df['wm_gm_mean'] = (df['age_pred_wm_age_nonlinear'] + df['age_pred_gm_age_ours']) / 2
    for cat in ['CN*', 'AD', 'MCI', 'CN']:
        q1 = df.loc[df['category']==cat, 'wm_gm_diff'].quantile(0.25)
        q3 = df.loc[df['category']==cat, 'wm_gm_diff'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_indices = df.loc[(df['category'] == cat) &
            ((df['wm_gm_diff'] < lower_bound) | (df['wm_gm_diff'] > upper_bound))].index
        df = df.drop(outlier_indices)
        print(f"Removed {len(outlier_indices)} outliers from {cat}.")
    
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
                # In the future, if we have more data points, we can apply the following criteria:
                # if row_c['dataset'] != row['dataset']:
                #     continue
                
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


def visualize_matched_data(df_matched, png):
    # hyperparameters for plotting
    xlim = [57, 96]
    xticks = [60,70,80,90]
    ylim_hist = [0, 13]
    ylim_ba = [-9, 8]
    yticks_hist = [0, 4, 8, 12]
    yticks_ba = [-8, -6, -4, -2, 0, 2, 4, 6, 8]
    fontsize = 9
    fontfamily = 'DejaVu Sans'
    palette_hist = {'male': 'tab:blue', 'female': 'tab:red'}
    marker_size = 5
    marker_linewidth = 0
    alpha = {
        'scatter': 0.75,
        'kde': 0.6,
    }
    color_hline_ba = 'limegreen'

    fig = plt.figure(figsize=(6.5, 6.5))
    gs = gridspec.GridSpec(nrows=5, ncols=4, wspace=0, hspace=0.5, height_ratios=[1, 0.05, 2, 0.05, 1])

    # histograms
    for i, cat in enumerate(['CN', 'CN*', 'MCI', 'AD']):
        ax = fig.add_subplot(gs[0, i])
        data = df_matched.loc[df_matched['category']==cat, ]
        sns.histplot(data=data, x='age', hue='sex', palette=palette_hist, alpha=1, binwidth=1, multiple='stack', ax=ax, legend=False)
        ax.set_title(f"{cat} (N={data.shape[0]})", fontsize=fontsize, fontfamily=fontfamily)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim_hist[0], ylim_hist[1])
        ax.set_xlabel('')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        ax.tick_params(axis='x', which='both', direction='out', length=4, pad=2, labelsize=fontsize, labelfontfamily=fontfamily)

        if i == 0:
            ax.set_ylabel('count', fontsize=fontsize, fontfamily=fontfamily)
            ax.set_yticks(yticks_hist)
            ax.set_yticklabels(yticks_hist)
        else:
            ax.set_ylabel('')
            ax.set_yticks(yticks_hist)
            ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', direction='inout', length=2)

    # xlabel and legend for histograms
    ax = fig.add_subplot(gs[1, :])
    ax.text(0.5, 0.9, 'chronological age (years)', fontsize=fontsize, fontfamily=fontfamily, ha='center', va='center', transform=ax.transAxes)
    patch1 = mpatches.Patch(edgecolor='black', facecolor=palette_hist['male'], linewidth=0.5, label='Male')
    patch2 = mpatches.Patch(edgecolor='black', facecolor=palette_hist['female'], linewidth=0.5, label='Female')
    ax.legend(handles=[patch1, patch2], loc='upper center', bbox_to_anchor=(0.5, 0.5), frameon=False, ncol=2, prop={'family': fontfamily, 'size':fontsize})
    ax.axis('off')

    # Bland-Altman plots
    mean_shift = df_matched.loc[df_matched['category']=='CN', 'wm_gm_diff'].mean()
    
    for i, cat in enumerate(['CN', 'CN*', 'MCI', 'AD']):
        ax = fig.add_subplot(gs[2, i])
        data = df_matched.loc[df_matched['category']==cat, ].copy()
        data['wm_gm_diff'] = data['wm_gm_diff'] - mean_shift
        
        # mean and std
        y_mean = data['wm_gm_diff'].mean()
        y_std = data['wm_gm_diff'].std()
        
        # p-value from Wilcoxon signed-rank test
        res = wilcoxon(x=data['wm_gm_diff'])
        pvalue = res.pvalue
        if pvalue > 0.001:
            text_pvalue = f"p-value≈{pvalue:.3f}"
        elif pvalue <= 0.001 and pvalue > 0.0001:
            text_pvalue = "p-value<0.001"
        elif pvalue < 0.0001:
            text_pvalue = "p-value≪0.001"
        else:
            text_pvalue = "p-value=???"
        
        sns.scatterplot(data=data, x='wm_gm_mean', y='wm_gm_diff',
            s=marker_size, linewidth=marker_linewidth, color='tab:red', alpha=alpha['scatter'],
            ax=ax)
        sns.kdeplot(data=data, x='wm_gm_mean', y='wm_gm_diff',
            fill=True, levels=10, cut=1, cmap='RdPu', alpha=alpha['kde'],
            ax=ax)  # rocket_r, PuRd, RdPu
        ax.axhline(y=0, linestyle='-', linewidth=1, color='k', alpha=0.25)
        ax.axhline(y=y_mean, linestyle='-', linewidth=1, color=color_hline_ba)
        ax.axhline(y=y_mean-1.96*y_std, linestyle='--', linewidth=1, color=color_hline_ba)
        ax.axhline(y=y_mean+1.96*y_std, linestyle='--', linewidth=1, color=color_hline_ba)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim_ba[0], ylim_ba[1])
        ax.text(0.04, 0.97, cat, fontsize=fontsize, fontfamily=fontfamily,
            transform=ax.transAxes, va='top', ha='left')
        ax.text(0.5, 0.01, text_pvalue, fontsize=fontsize, fontfamily=fontfamily,
            transform=ax.transAxes, ha='center', va='bottom')
        ax.set_xlabel('')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        ax.tick_params(axis='x', which='both', direction='out', length=4, pad=2, labelsize=fontsize, labelfontfamily=fontfamily)

        if i == 0:
            ax.set_ylabel('WMage - GMage (years, CN adjusted)', fontsize=fontsize, fontfamily=fontfamily)
            ax.set_yticks(yticks_ba)
            ax.set_yticklabels(yticks_ba)
        else:
            ax.set_ylabel('')
            ax.set_yticks(yticks_ba)
            ax.set_yticklabels([])
        ax.tick_params(axis='y', which='both', direction='inout', length=2)
        
    # xlabel and legend for Bland-Altman plots
    ax = fig.add_subplot(gs[3, :])
    ax.text(0.5, 0.9, f"(WMage + GMage) / 2 (years)", fontsize=fontsize, fontfamily=fontfamily, ha='center', va='center', transform=ax.transAxes)
    line1 = mlines.Line2D([], [], color=color_hline_ba, linestyle='-', linewidth=1, label='Mean')
    line2 = mlines.Line2D([], [], color=color_hline_ba, linestyle='--', linewidth=1, label='Mean ± 1.96 SD')
    ax.legend(handles=[line1, line2], loc='upper center', bbox_to_anchor=(0.5, 0.5), frameon=False, ncol=2, prop={'family': fontfamily, 'size':fontsize})
    ax.axis('off')

    # time to MCI for CN* subjects
    data = df_matched.loc[df_matched['category']=='CN*', ]
    ax = fig.add_subplot(gs[4, :])
    
    ax = sns.violinplot(
        data=data, x='time_to_MCI', orient='h', 
        color='tan', width=0.6, linewidth=0, inner=None, cut=0, density_norm='count',
        ax=ax)

    for item in ax.collections:
        x0, y0, width, height = item.get_paths()[0].get_extents().bounds
        item.set_clip_path(plt.Rectangle((x0, y0), width, height/2, transform=ax.transData))
    
    num_items = len(ax.collections)
    ax = sns.stripplot(
        data=data, x='time_to_MCI', orient='h',
        color='tan', jitter=0.15, alpha=0.5, size=2, 
        ax=ax)

    for item in ax.collections[num_items:]:
        item.set_offsets(item.get_offsets() + (0,0.25))

    ax = sns.boxplot(
        data=data, x='time_to_MCI', orient='h',
        width=0.1, linecolor='black', showfliers=False, boxprops=dict(facecolor=(0,0,0,0), linewidth=1, zorder=2),
        whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1), medianprops=dict(linewidth=1.5, color='gold'),
        ax=ax)

    ax.text(0.02, 0.97, f'{data["time_to_MCI"].mean():.1f} ± {data["time_to_MCI"].std():.1f} years ({data["time_to_MCI"].min():.1f} - {data["time_to_MCI"].max():.1f} years)', 
        ha='left', va='top',
        fontsize=9, fontfamily='DejaVu Sans',
        transform=ax.transAxes, verticalalignment='top')
    
    ax.set_xlim(-0.5, 7.5)
    ax.invert_xaxis()
    ax.tick_params(axis='x', which='both', direction='out', length=4, pad=2, labelsize=fontsize, labelfontfamily=fontfamily)
    ax.set_xlabel('time to MCI for CN* participants (years)', fontsize=fontsize, fontfamily=fontfamily)
    ax.set_ylabel('CN*', fontsize=fontsize, fontfamily=fontfamily)
    ax.tick_params(axis='y', which='both', direction='out', length=4)
    
    # Save figure
    fig.savefig(png, dpi=600)

if __name__ == '__main__':
    # Load data
    output_fn = 'experiments/2024-05-07_Matched_Cohort_Linear_Model/data/data_prep.csv'
    if Path(output_fn).is_file():
        df = pd.read_csv(output_fn)
    else:
        data_prep = DataPreparation(roster_brain_age_models(), '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv')
        df = data_prep.load_predictions_of_all_models()
        df = data_prep.retrieve_diagnosis_label(df)
        df = data_prep.assign_cn_label(df)
        df = data_prep.mark_progression_subjects_out(df)
        df.to_csv(output_fn, index=False)
        
    # Matched categories of data points
    cnstar_threshold_choices = [(0,10), (1,10), (2,10), 
        (0,6), (1,6), (2,6), (0,5), (1,5), (2,5), (0,4), (1,4), (2,4)]
    for t in cnstar_threshold_choices:
        df_matched = get_matched_cohort(df, cnstar_threshold=t, age_diff_threshold=1)
        df_matched.to_csv(f'experiments/2024-05-07_Matched_Cohort_Linear_Model/data/df_matched_cnstar-{t[0]}-{t[1]}.csv', index=False)
        visualize_matched_data(df_matched, png=f'experiments/2024-05-07_Matched_Cohort_Linear_Model/figs/blant_altman_matched_cohort_cnstar-{t[0]}-{t[1]}_v1.png')