import pickle
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

# Organize and merge results from two experiments
dict_results_merged = {'Global Model': {}, 'Time-Specific Model': {}}

# Bootstrapped AUCs
dict_results_merged['Global Model']['df_aucs'] = pd.read_csv('reports/figures/2024-05-13_MCI_Prediction_From_T_Minus_N_Years_Two_Methods_Merged/data/one_model/prediction_auc_bootstrap_w-bc_MCI_age-0-1000_hungry_but_picky_match-wo-dataset_ws-1.csv')
dict_results_merged['Time-Specific Model']['df_aucs'] = pd.read_csv('reports/figures/2024-05-13_MCI_Prediction_From_T_Minus_N_Years_Two_Methods_Merged/data/time_specific/prediction_auc_bootstrap_w-bc_age-0-1000_MCI_ws-1_hungry_but_picky_match-wo-dataset.csv')

# Windowed data subsets
with open('reports/figures/2024-05-13_MCI_Prediction_From_T_Minus_N_Years_Two_Methods_Merged/data/one_model/dict_windowed_results_one_model.pkl', 'rb') as f:
    dict_results = pickle.load(f)
data_subsets = {'idx': [], 'time_to_MCI': [], 'num_pairs': []}
for idx, w_results in dict_results.items():
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
    w_results = subset['data_matched']
    w_results = w_results.loc[w_results['time_to_MCI']>=0, ].drop_duplicates(subset=['match_id'], keep='first', ignore_index=True)
    data_subsets['idx'] += [idx]*len(w_results.index)
    data_subsets['time_to_MCI'] += w_results['time_to_MCI'].values.tolist()
    data_subsets['num_pairs'] += [w_results['match_id'].nunique()] * len(w_results.index)
data_subsets = pd.DataFrame(data_subsets)
dict_results_merged['Time-Specific Model']['data_subsets'] = data_subsets

# Figure hyperparameters
fontsize = 9
fontsize_legend = 8
fontfamily = 'DejaVu Sans'
linewidth = 2
ylim = [0.35, 0.85]
y_ticks = [0.4, 0.5, 0.6, 0.7, 0.8]

fig = plt.figure(figsize=(6.5, 8), tight_layout=True)
gs = gridspec.GridSpec(nrows=6, ncols=3, wspace=0, hspace=0, width_ratios=[0.3, 3.1, 3.1],  height_ratios=[0.3, 1.54, 1.54, 1.54, 1.54, 1.54])
clf_names = ['Logistic Regression','Linear SVM','Random Forest']
dict_feat_combos = {
    'basic: chronological age + sex': {'color': (0,0,0), 'alpha': 1, 'linestyle': 'solid'},
    'basic + WM age nonlinear': {'color': (0,0,1), 'alpha': 1, 'linestyle': 'solid'},
    'basic + WM age affine': {'color': (0.5,0,1), 'alpha': 1, 'linestyle': 'solid'},
    'basic + GM age (ours)': {'color': (1,0,0), 'alpha': 1, 'linestyle': 'solid'},
    'basic + GM age (DeepBrainNet)': {'color': (1,0.5,0.5), 'alpha': 1, 'linestyle': 'dashed'},
    'basic + GM age (TSAN)': {'color': (0.5,0,0), 'alpha': 1, 'linestyle': 'dotted'},
    'basic + WM age nonlinear + GM age (ours)': {'color': (0,1,0), 'alpha': 1, 'linestyle': 'solid'},
    'basic + WM age nonlinear + GM age (DeepBrainNet)': {'color': (0.5,1,0.5), 'alpha': 1, 'linestyle': 'dashed'},        
    'basic + WM age nonlinear + GM age (TSAN)': {'color': (0,0.5,0), 'alpha': 1, 'linestyle': 'dotted'},        
}
feat_vis_order = [
    'basic: chronological age + sex',
    'basic + WM age nonlinear + GM age (TSAN)', 'basic + WM age nonlinear + GM age (DeepBrainNet)',
    'basic + WM age nonlinear + GM age (ours)',
    'basic + GM age (TSAN)', 'basic + GM age (DeepBrainNet)', 'basic + GM age (ours)',
    'basic + WM age affine', 'basic + WM age nonlinear',
]

# Bottom block: legends
ax = fig.add_subplot(gs[5,:])
lines = []
for feat_combo in dict_feat_combos.keys():
    label_txt = textwrap.fill(feat_combo, width=25)
    
    line = mlines.Line2D(
        [], [], color=dict_feat_combos[feat_combo]['color'], alpha=dict_feat_combos[feat_combo]['alpha'], 
        linestyle=dict_feat_combos[feat_combo]['linestyle'], linewidth=linewidth, label=label_txt)
    lines.append(line)
ax.legend(
    handles=lines, loc='lower right', bbox_to_anchor=(1, 0), 
    prop={'size':fontsize_legend, 'family':fontfamily}, 
    borderpad=0.5, labelspacing=1.5, handlelength=2.3, handletextpad=0.5, borderaxespad=0, columnspacing=1,
    ncols=3,
    frameon=True)
ax.axis('off')

# Left two blocks: y axis labels
ax = fig.add_subplot(gs[1:4,0])
ax.text(
    0, 0.5, 'Area Under the ROC Curve for MCI Prediction', fontsize=fontsize, fontfamily=fontfamily,
    ha='right', va='center', rotation='vertical', transform=ax.transAxes
    )
ax.axis('off')
ax = fig.add_subplot(gs[4,0])
ax.text(
    0, 0.5, 'Subsets from Sliding Windows', fontsize=fontsize, fontfamily=fontfamily,
    ha='right', va='center', rotation='vertical', transform=ax.transAxes
    )
ax.axis('off')

# Middle and right blocks: two methods
for i, method in enumerate(['Global Model', 'Time-Specific Model']):
    df_aucs = dict_results_merged[method]['df_aucs']
    data_subsets = dict_results_merged[method]['data_subsets']
    xlim= [-0.25, df_aucs['time_to_MCI'].max()+0.5]

    # title
    ax = fig.add_subplot(gs[0,i+1])
    ax.text(0.5, 0.5, method, fontsize=fontsize, fontfamily=fontfamily, ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')

    # AUCs
    for j, classifier in enumerate(clf_names):
        ax = fig.add_subplot(gs[j+1,i+1])
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
            
        ax.vlines(x=df_aucs['time_to_MCI'].unique(), ymin=0, ymax=1, transform=ax.get_xaxis_transform(), color=(0,0,0), linestyle='-', linewidth=1, alpha=0.1)
        ax.text(0.02, 0.95, classifier, fontsize=fontsize, fontfamily=fontfamily, transform=ax.transAxes, verticalalignment='top')
        ax.set_xlim(left=xlim[0], right=xlim[1])
        ax.set_xticks([])
        ax.invert_xaxis()
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
        if i==0:
            ax.set_yticks(y_ticks)
            ax.tick_params(axis='y', which='both', direction='inout', length=2, pad=1, labelsize=fontsize, labelfontfamily=fontfamily)
        else:
            ax.set_yticks([])
        ax.set_ylabel('')

    # time to event distribution for each subset
    ax = fig.add_subplot(gs[4,i+1])
    sns.violinplot(data=data_subsets, x='time_to_MCI', y='idx', orient='h', color='lightgray', width=2, linewidth=1, split=True, inner=None, cut=0, density_norm='count', native_scale=True, ax=ax)
    ax.vlines(x=df_aucs['time_to_MCI'].unique(), ymin=0, ymax=1, transform=ax.get_xaxis_transform(), color=(0,0,0), linestyle='-', linewidth=1, alpha=0.1)
    for idx in data_subsets['idx'].unique():
        num = data_subsets.loc[data_subsets['idx']==idx, 'num_pairs'].values[0]
        ax.text(data_subsets.loc[data_subsets['idx']==idx, 'time_to_MCI'].mean(), idx+1, f'{num}', fontsize=fontsize*0.75, fontfamily=fontfamily, ha='center', va='center')
    x_ticks = np.arange(0, df_aucs['time_to_MCI'].max()+1, 1)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', which='both', direction='out', length=2, pad=1, labelsize=fontsize, labelfontfamily=fontfamily)
    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.set_xlabel(f"{'time_to_MCI'.replace('time_to_', 'Time to ')} (years)", fontsize=fontsize, fontfamily=fontfamily, labelpad=1)

fig.savefig('reports/figures/2024-05-13_MCI_Prediction_From_T_Minus_N_Years_Two_Methods_Merged/figs/figure_sliding_window_auc_brain_age_features.png', dpi=600)
