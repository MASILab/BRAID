# After cohort matching, visualize i) the distribution of the chronological age and sex,
# and ii) the Bland-Altman plot of the WMage and GMage for each category.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.stats import wilcoxon

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

df = pd.read_csv('experiments/2024-03-20_Matched_Cohort_Linear_Model/data_matched_cohort.csv')

fig = plt.figure(figsize=(6.5, 5), tight_layout=True)
gs = gridspec.GridSpec(4, 4, wspace=0, hspace=0.5, height_ratios=[1, 0.05, 2, 0.05])

# histograms
for i, cat in enumerate(['CN', 'CN*', 'MCI', 'AD']):
    data = df.loc[df['category_criteria_1']==cat, ]
    ax = fig.add_subplot(gs[0, i])
    sns.histplot(data=data, x='age', hue='sex', palette=palette_hist, alpha=1, binwidth=1, multiple='stack', ax=ax, legend=False)
    ax.set_title(f"{cat} (N={data.shape[0]})", fontsize=fontsize, fontfamily=fontfamily)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim_hist[0], ylim_hist[1])
    ax.set_xlabel('')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=fontsize, fontfamily=fontfamily)

    if i == 0:
        ax.set_ylabel('count', fontsize=fontsize, fontfamily=fontfamily)
        ax.set_yticks(yticks_hist)
        ax.set_yticklabels(yticks_hist, fontsize=fontsize, fontfamily=fontfamily)
    else:
        ax.set_ylabel('')
        ax.set_yticks(yticks_hist)
        ax.set_yticklabels([])

# xlabel and legend for histograms
ax = fig.add_subplot(gs[1, :])
ax.text(0.5, 0.9, 'chronological age (years)', fontsize=fontsize, fontfamily=fontfamily, ha='center', va='center', transform=ax.transAxes)
patch1 = mpatches.Patch(edgecolor='black', facecolor=palette_hist['male'], label='Male')
patch2 = mpatches.Patch(edgecolor='black', facecolor=palette_hist['female'], label='Female')
ax.legend(handles=[patch1, patch2], loc='upper center', fontsize=fontsize, frameon=False, ncol=2, bbox_to_anchor=(0.5, 0.5))
ax.axis('off')

# Bland-Altman plots
for i, cat in enumerate(['CN', 'CN*', 'MCI', 'AD']):
    data = df.loc[df['category_criteria_1']==cat, ]
    ax = fig.add_subplot(gs[2, i])
    
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
        s=marker_size, linewidth=marker_linewidth, 
        color='tab:red', alpha=alpha['scatter'],
        ax=ax,
    )
    sns.kdeplot(data=data, x='wm_gm_mean', y='wm_gm_diff',
        fill=True, levels=10, cut=1,
        cmap='RdPu', alpha=alpha['kde'],
        ax=ax,
    )  # rocket_r, PuRd, RdPu
    ax.axhline(y=0, linestyle='-', linewidth=1, color='k', alpha=0.25)
    ax.axhline(y=y_mean, linestyle='-', linewidth=1, color=color_hline_ba)
    ax.axhline(y=y_mean-1.96*y_std, linestyle='--', linewidth=1, color=color_hline_ba)
    ax.axhline(y=y_mean+1.96*y_std, linestyle='--', linewidth=1, color=color_hline_ba)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim_ba[0], ylim_ba[1])
    ax.text(0.05, 0.95, cat, fontsize=fontsize, fontfamily=fontfamily,
        transform=ax.transAxes, verticalalignment='top')
    ax.text(0.05, 0.08, text_pvalue, fontsize=fontsize, fontfamily=fontfamily,
        transform=ax.transAxes, verticalalignment='top')
    ax.set_xlabel('')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=fontsize, fontfamily=fontfamily)

    if i == 0:
        ax.set_ylabel('WMage - GMage (years)', fontsize=fontsize, fontfamily=fontfamily)
        ax.set_yticks(yticks_ba)
        ax.set_yticklabels(yticks_ba, fontsize=fontsize, fontfamily=fontfamily)
    else:
        ax.set_ylabel('')
        ax.set_yticks(yticks_ba)
        ax.set_yticklabels([])

# xlabel and legend for Bland-Altman plots
ax = fig.add_subplot(gs[3, :])
ax.text(0.5, 0.9, f"(WMage + GMage) / 2 (years)", fontsize=fontsize, fontfamily=fontfamily, ha='center', va='center', transform=ax.transAxes)
line1 = mlines.Line2D([], [], color=color_hline_ba, linestyle='-', linewidth=1, label='Mean')
line2 = mlines.Line2D([], [], color=color_hline_ba, linestyle='--', linewidth=1, label='Mean ± 1.96 SD')
ax.legend(handles=[line1, line2], loc='upper center', fontsize=fontsize, frameon=False, ncol=2, bbox_to_anchor=(0.5, 0.5))
ax.axis('off')

# # Save figure
fig.savefig('experiments/2024-03-20_Matched_Cohort_Linear_Model/figs/blant_altman_matched_cohort_v4.png', dpi=600)