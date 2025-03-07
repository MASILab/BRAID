# Based on the same data as 
# 2024-05-13_MCI_Prediction_From_T_Minus_N_Years_Two_Methods_Merged
# but optimized the layout.

import pdb
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Data
dict_results_merged = {'Global Model': {}, 'Time-Specific Models': {}}

# Load AUC results and exclude subsets with <= 10 pairs
df = pd.read_csv('reports/figures/2024-05-13_MCI_Prediction_From_T_Minus_N_Years_Two_Methods_Merged/data/one_model/prediction_auc_bootstrap_w-bc_MCI_age-0-1000_hungry_but_picky_match-wo-dataset_ws-1.csv')
df = df.loc[df['idx']<=12, ]
dict_results_merged['Global Model']['df_aucs'] = df.copy()

df = pd.read_csv('reports/figures/2024-05-13_MCI_Prediction_From_T_Minus_N_Years_Two_Methods_Merged/data/time_specific/prediction_auc_bootstrap_w-bc_age-0-1000_MCI_ws-1_hungry_but_picky_match-wo-dataset.csv')
df = df.loc[df['idx']<=12, ]
dict_results_merged['Time-Specific Models']['df_aucs'] = df.copy()

# load subsets information and exclude subsets with <= 10 pairs
with open('reports/figures/2024-05-13_MCI_Prediction_From_T_Minus_N_Years_Two_Methods_Merged/data/one_model/dict_windowed_results_one_model.pkl', 'rb') as f:
    dict_results = pickle.load(f)
data_subsets = {'idx': [], 'time_to_MCI': [], 'num_pairs': []}
for idx, w_results in dict_results.items():
    if idx > 12:
        continue
    w_results = w_results.loc[w_results['time_to_MCI']>=0, ].drop_duplicates(subset=['match_id'], keep='first', ignore_index=True)
    data_subsets['idx'] += [idx]*len(w_results.index)
    data_subsets['time_to_MCI'] += w_results['time_to_MCI'].values.tolist()
    data_subsets['num_pairs'] += [w_results['match_id'].nunique()] * len(w_results.index)
data_subsets = pd.DataFrame(data_subsets)
dict_results_merged['Global Model']['data_subsets'] = data_subsets

with open('reports/figures/2024-05-13_MCI_Prediction_From_T_Minus_N_Years_Two_Methods_Merged/data/time_specific/dict_windowed_results_time_specific_model.pkl', 'rb') as f:
    dict_results = pickle.load(f)
data_subsets = {'idx': [], 'time_to_MCI': [], 'num_pairs': []}
for idx, subset in dict_results.items():
    if idx > 12:
        continue
    w_results = subset['data_matched']
    w_results = w_results.loc[w_results['time_to_MCI']>=0, ].drop_duplicates(subset=['match_id'], keep='first', ignore_index=True)
    data_subsets['idx'] += [idx]*len(w_results.index)
    data_subsets['time_to_MCI'] += w_results['time_to_MCI'].values.tolist()
    data_subsets['num_pairs'] += [w_results['match_id'].nunique()] * len(w_results.index)
data_subsets = pd.DataFrame(data_subsets)
dict_results_merged['Time-Specific Models']['data_subsets'] = data_subsets

# Figure hyperparameters
fontsize = 9
fontsize_legend = 8
fontfamily = 'DejaVu Sans'
linewidth = 1
ylim = [0.25, 0.85]
y_ticks = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
clf_names = ['Logistic Regression','Linear SVM','Random Forest']
dict_feat_combos = {
    'basic: chronological age + sex': {'color': (0,0,0), 'alpha': 1, 'linestyle': 'solid'},
    'basic + WM age nonlinear': {'color': (0,0,1), 'alpha': 1, 'linestyle': 'solid'},
    'basic + WM age affine': {'color': (0.5,0,1), 'alpha': 1, 'linestyle': 'solid'},
    'basic + GM age (ours)': {'color': (1,0,0), 'alpha': 1, 'linestyle': 'solid'},
    'basic + GM age (DeepBrainNet)': {'color': (1,0.5,0.5), 'alpha': 1, 'linestyle': 'dashed'},
    'basic + GM age (TSAN)': {'color': (0.5,0,0), 'alpha': 1, 'linestyle': 'dotted'},
    'basic + WM age nonlinear + GM age (ours)': {'color': (0,1,0), 'alpha': 1, 'linestyle': 'solid'},
    'basic + WM age nonlinear + GM age (DeepBrainNet)': {'color': (0.75,1,0.75), 'alpha': 1, 'linestyle': 'dashed'},        
    'basic + WM age nonlinear + GM age (TSAN)': {'color': (0,0.5,0), 'alpha': 1, 'linestyle': 'dotted'},        
}
feat_vis_order = [
    'basic: chronological age + sex',
    'basic + WM age nonlinear + GM age (TSAN)', 'basic + WM age nonlinear + GM age (DeepBrainNet)',
    'basic + WM age nonlinear + GM age (ours)',
    'basic + GM age (TSAN)', 'basic + GM age (DeepBrainNet)', 'basic + GM age (ours)',
    'basic + WM age affine', 'basic + WM age nonlinear',
]
dict_feat_combos_rename = {
    'basic: chronological age + sex': 'basic: chronological age + sex',
    'basic + WM age nonlinear': 'basic + WM age nonrigid',
    'basic + WM age affine': 'basic + WM age affine',
    'basic + GM age (ours)': 'basic + GM age (ours)',
    'basic + GM age (DeepBrainNet)': 'basic + GM age (DBN)',
    'basic + GM age (TSAN)': 'basic + GM age (TSAN)',
    'basic + WM age nonlinear + GM age (ours)': 'basic + WM age nonrigid + GM age (ours)',
    'basic + WM age nonlinear + GM age (DeepBrainNet)': 'basic + WM age nonrigid + GM age (DBN)',
    'basic + WM age nonlinear + GM age (TSAN)': 'basic + WM age nonrigid + GM age (TSAN)',
}

for _, method in enumerate(['Global Model', 'Time-Specific Models']):

    fig = plt.figure(figsize=(3, 4), tight_layout=True)
    gs = gridspec.GridSpec(nrows=4, ncols=1, hspace=0)

    df_aucs = dict_results_merged[method]['df_aucs']
    data_subsets = dict_results_merged[method]['data_subsets']
    xlim= [-0.25, df_aucs['time_to_MCI'].max()+0.5]

    # subplot 1: time to event distribution for each subset
    ax = fig.add_subplot(gs[0])
    sns.violinplot(
        data=data_subsets, x='time_to_MCI', y='idx', orient='h', 
        color='#ffbf00', edgecolor='#06b050', width=5, linewidth=0.5, 
        split=True, inner=None, cut=0, 
        density_norm='count', native_scale=True, 
        ax=ax, 
        )
    ax.vlines(
        x=df_aucs['time_to_MCI'].unique(), ymin=0, ymax=1, 
        transform=ax.get_xaxis_transform(), 
        color=(0,0,0), 
        linestyle='-', linewidth=1, alpha=0.1
        )
    for idx in data_subsets['idx'].unique():
        num = data_subsets.loc[data_subsets['idx']==idx, 'num_pairs'].values[0]
        ax.text(data_subsets.loc[data_subsets['idx']==idx, 'time_to_MCI'].mean(), idx+1, f'{num}', fontsize=8, fontfamily=fontfamily, ha='center', va='center')
    
    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.set_yticks([])
    ax.set_ylabel('')

    # AUCs
    for i, classifier in enumerate(clf_names):
        ax = fig.add_subplot(gs[i+1])
        for feat_combo in feat_vis_order:
            data = df_aucs.loc[(df_aucs['clf_name']==classifier)&(df_aucs['feat_combo_name']==feat_combo), ].copy()
            ax.plot(
                data['time_to_MCI'].values, data['auc_mean'].values,
                linewidth=linewidth, linestyle=dict_feat_combos[feat_combo]['linestyle'],
                color=dict_feat_combos[feat_combo]['color'], alpha=dict_feat_combos[feat_combo]['alpha'],
                )
            ax.fill_between(
                x=data['time_to_MCI'].values, y1=data['auc_upper'].values, y2=data['auc_lower'].values,
                color=dict_feat_combos[feat_combo]['color'], alpha=dict_feat_combos[feat_combo]['alpha']*0.1, linewidth=0)
            
        ax.vlines(
            x=df_aucs['time_to_MCI'].unique(), ymin=0, ymax=1, 
            transform=ax.get_xaxis_transform(), 
            color=(0,0,0), 
            linestyle='-', linewidth=1, alpha=0.1
            )
        ax.text(0.02, 0.95, classifier, fontsize=fontsize, fontfamily=fontfamily, transform=ax.transAxes, verticalalignment='top')
        
        ax.set_xlim(left=xlim[0], right=xlim[1])
        ax.invert_xaxis()
        ax.set_xticks([])
        ax.set_xlabel('')
        
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='y', which='both', direction='inout', length=2, pad=1, labelsize=fontsize, labelfontfamily=fontfamily)
        ax.set_ylabel('')
    
    x_ticks = np.arange(0, 7, 1)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', which='both', direction='out', length=2, pad=1, labelsize=fontsize, labelfontfamily=fontfamily)
    ax.set_xlabel(f"{'time_to_MCI'.replace('time_to_', 'Time to ')} (years)", fontsize=fontsize, fontfamily=fontfamily, labelpad=1)
    
    fig.subplots_adjust(hspace=0)
    fig.savefig(f'reports/figures/2025-01-21_MCI_Prediction_From_T_Minus_N_Years_Two_Methods_Parts/figs/{method}.png', dpi=300)
