import pdb
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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

    def take_cross_sectional_samples(self, df, age_range=(65,75), random_seed=0):
        df['subj'] = df['dataset'] + '_' + df['subject']
        df['category'] = None
        filter_age = (df['age_gt']>=age_range[0]) & (df['age_gt']<age_range[1])
        
        # CN2MCI (current CN, later MCI), CN2AD (current CN, later AD), MCI2AD (current MCI, later AD)
        for subj in df.loc[filter_age, 'subj'].unique():
            rows_subj = df.loc[df['subj']==subj, ].copy()
            rows_subj = rows_subj.sort_values(by='age_gt')
            for i in range(len(rows_subj.index)-1):
                if (rows_subj.iloc[i]['diagnosis'] == 'normal') & (rows_subj.iloc[i+1]['diagnosis'] == 'MCI'):
                    df.loc[(df['subj'] == subj) & (df['age_gt'] == rows_subj.iloc[i]['age_gt']), 'category'] = 'CN2MCI'
                elif (rows_subj.iloc[i]['diagnosis'] == 'normal') & (rows_subj.iloc[i+1]['diagnosis'] == 'dementia'):
                    df.loc[(df['subj'] == subj) & (df['age_gt'] == rows_subj.iloc[i]['age_gt']), 'category'] = 'CN2AD'
                elif (rows_subj.iloc[i]['diagnosis'] == 'MCI') & (rows_subj.iloc[i+1]['diagnosis'] == 'dementia'):
                    df.loc[(df['subj'] == subj) & (df['age_gt'] == rows_subj.iloc[i]['age_gt']), 'category'] = 'MCI2AD'
        # CN
        for subj in df.loc[filter_age & (df['diagnosis']=='normal'), 'subj'].unique():
            if subj in df.loc[df['category'].notna(), 'subj'].unique():
                continue
            if len(df.loc[df['subj']==subj, 'diagnosis'].unique()) == 1:
                sampled_row = df.loc[filter_age & (df['subj']==subj),].sample(n=1, random_state=random_seed)
                df.loc[sampled_row.index, 'category'] = 'CN'
        # MCI
        for subj in df.loc[filter_age & (df['diagnosis']=='MCI'), 'subj'].unique():
            if subj in df.loc[df['category'].notna(), 'subj'].unique():
                continue
            sampled_row = df.loc[filter_age & (df['subj']==subj) & (df['diagnosis']=='MCI'),].sample(n=1, random_state=random_seed)
            df.loc[sampled_row.index, 'category'] = 'MCI'
        # AD
        for subj in df.loc[filter_age & (df['diagnosis']=='dementia'), 'subj'].unique():
            if subj in df.loc[df['category'].notna(), 'subj'].unique():
                continue
            sampled_row = df.loc[filter_age & (df['subj']==subj) & (df['diagnosis']=='dementia'),].sample(n=1, random_state=random_seed)
            df.loc[sampled_row.index, 'category'] = 'AD'
        
        df = df.loc[filter_age & df['category'].notna(),]
        return df

dict_models = {
    'WM age model': {
        'prediction_root': 'models/2024-02-07_ResNet101_BRAID_warp/predictions',
        'col_suffix': '_wm_age',
        },
    'WM age model (contaminated with GM age features)': {
        'prediction_root': 'models/2023-12-22_ResNet101/predictions',
        'col_suffix': '_wm_age_contaminated',
        },
    'GM age model (ours)': {
        'prediction_root': 'models/2024-02-07_T1wAge_ResNet101/predictions',
        'col_suffix': '_gm_age_ours',
        },
    'GM age model (TSAN)': {
        'prediction_root': 'models/2024-02-12_TSAN_first_stage/predictions',
        'col_suffix': '_gm_age_tsan',
        },
}

d = DataPreparation(dict_models, databank_csv='/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv')
df = d.load_data()
# # Make sure that the mean of age_gt of each category is similar 
# for seed in range(100):
#     df_cs = d.take_cross_sectional_samples(df, age_range=(65,75), random_seed=seed)  
#     m1 = df_cs.loc[df_cs['category']=='CN', 'age_gt'].mean()
#     m2 = df_cs.loc[df_cs['category']=='MCI', 'age_gt'].mean()
#     m3 = df_cs.loc[df_cs['category']=='AD', 'age_gt'].mean()
#     print(f"seed: {seed}\t {m1} {m2} {m3}")
df = d.take_cross_sectional_samples(df, age_range=(65,75), random_seed=5)
df = df.melt(
    id_vars=['dataset','subject','session','age_gt','category'],
    value_vars=[f'age_pred{dict_models[model]["col_suffix"]}_mean' for model in dict_models.keys()], 
    var_name='age_pred_name', 
    value_name='age_pred_mean')
df['BAG'] = df['age_pred_mean'] - df['age_gt']
df['category_coarse'] = df['category'].map({'CN':'CN', 'MCI':'MCI', 'AD':'AD', 'CN2MCI':'CN', 'MCI2AD':'MCI'})
dict_category_center = {'CN': 1.5, 'MCI': 4.5, 'AD': 7.5}
dict_method_shift = {
    'age_pred_wm_age_mean': -0.9,
    'age_pred_wm_age_contaminated_mean': -0.3,
    'age_pred_gm_age_ours_mean': 0.3,
    'age_pred_gm_age_tsan_mean': 0.9,
}
df['x_position'] = df.apply(lambda row: dict_category_center[row['category_coarse']] + dict_method_shift[row['age_pred_name']], axis=1)
dict_colors = {
    'age_pred_wm_age_mean': 'tab:blue',
    'age_pred_wm_age_contaminated_mean': 'tab:purple',
    'age_pred_gm_age_ours_mean': 'tab:red',
    'age_pred_gm_age_tsan_mean': 'salmon',
}

fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))
ax = sns.violinplot(
    data = df.loc[df['category'].isin(['CN','MCI','AD']),],
    x = 'x_position',
    y = 'BAG',
    hue = 'age_pred_name',
    palette = dict_colors,
    cut=0,
    width=1,
    inner=None,
    ax=ax,
    saturation=1,
    linewidth=0,
    native_scale=True,
    )

for item in ax.collections:
    x0, y0, width, height = item.get_paths()[0].get_extents().bounds
    item.set_clip_path(plt.Rectangle((x0, y0), width/2, height, transform=ax.transData))
    
num_items = len(ax.collections)
ax = sns.stripplot(
    data = df.loc[df['category'].isin(['CN','MCI','AD']),],
    x='x_position',
    y= 'BAG',
    hue='age_pred_name',
    palette = dict_colors,
    jitter=0.15,
    alpha=0.4,
    size=2,
    ax=ax,
    legend=False,
    native_scale=True
    )

ax = sns.stripplot(
    data = df.loc[df['category'].isin(['CN2MCI','MCI2AD']),],
    x='x_position',
    y= 'BAG',
    color='lime',
    jitter=0.15,
    alpha=1,
    size=4,
    marker='*',
    ax=ax,
    legend=False,
    native_scale=True
    )

for item in ax.collections[num_items:]:
    item.set_offsets(item.get_offsets() + (0.15,0))

ax = sns.boxplot(
    data = df.loc[df['category'].isin(['CN','MCI','AD']),],
    x='x_position',
    y='BAG',
    width=0.2,
    linecolor='dimgrey',
    showfliers=False,
    boxprops=dict(facecolor=(0,0,0,0),
                  linewidth=1.5, zorder=2),
    whiskerprops=dict(linewidth=1),
    capprops=dict(linewidth=1.5),
    medianprops=dict(linewidth=1.5, color='black'),
    ax=ax,
    native_scale=True)

ax.axhline(y=0, linestyle='-',linewidth=1, color = 'black', alpha=0.5)
ax.set_xticks([1.5, 4.5, 7.5], minor=True)
ax.set_xticklabels(['CN', 'MCI', 'AD'], minor=True)
ax.set_xticks([3, 6], minor=False)
ax.set_xticklabels(['', ''], minor=False)
ax.tick_params(labelsize=9, which='both')
ax.grid(linestyle=':', linewidth=1, which='major')
ax.set_xlabel('Category', fontsize=9)
ax.set_ylabel('Brain Age Gap (year)', fontsize=9)

# legend
legend_scatter = mlines.Line2D(
    [], [], color='lime', marker='*', linestyle='None',
    markersize=4, label='progress to MCI/AD')

dict_label = {
    'age_pred_wm_age_mean': 'WM age',
    'age_pred_wm_age_contaminated_mean': 'WM age (contaminated)',
    'age_pred_gm_age_ours_mean': 'GM age (ours)',
    'age_pred_gm_age_tsan_mean': 'GM age (TSAN)',
}

handles, labels = ax.get_legend_handles_labels()
handles.append(legend_scatter)
labels.append('progressing to MCI/AD')

legend = ax.legend(handles, labels, fontsize=8.5)
for text in legend.texts:
    if text.get_text() in dict_label.keys():
        text.set_text(dict_label[text.get_text()])

fig.savefig('reports/figures/2024-03-12_Raincloud_plot_Diagnosis_Group_Comparison/figs/raincloud_plot_diagnosis_group_comparison_v2.png', dpi=300, bbox_inches='tight')