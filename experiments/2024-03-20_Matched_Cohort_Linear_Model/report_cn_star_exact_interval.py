# Specify the precise definition of CN* after the matching process.
# Draw raincloud plot to show the distribution of the interval between CN* and MCI/dementia diagnosis.

import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('experiments/2024-03-20_Matched_Cohort_Linear_Model/data_matched_cohort.csv')
subj_cn_star = df.loc[df['category_criteria_1']=='CN*', 'subj'].unique()

# age and diagnosis information of all sessions
df = pd.read_csv('models/2024-02-07_ResNet101_BRAID_warp/predictions/predicted_age_fold-1_test_bc.csv')
databank = pd.read_csv('/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv')
df['subj'] = df['dataset'] + '_' + df['subject']
df['diagnosis'] = None
    
for i, row in tqdm(df.iterrows(), total=len(df.index), desc='Retrieve diagnosis information'):
    loc_filter = (databank['dataset']==row['dataset']) & (databank['subject']==row['subject']) & ((databank['session']==row['session']) | (databank['session'].isnull()))
    if row['dataset'] in ['UKBB']:
        control_label = databank.loc[loc_filter, 'control_label'].values[0]
        df.loc[i, 'diagnosis'] = 'normal' if control_label == 1 else None
    else:
        df.loc[i, 'diagnosis'] = databank.loc[loc_filter, 'diagnosis_simple'].values[0]


# Calculate the interval between CN* and MCI/dementia diagnosis
list_intervals = []
for subj in subj_cn_star:
    rows_subj = df.loc[df['subj']==subj, ].copy()
    rows_subj = rows_subj.sort_values(by='age')
    for i in range(len(rows_subj.index)-1):
        if (rows_subj.iloc[i]['diagnosis']=='normal') & (rows_subj.iloc[i+1]['diagnosis'] in ['MCI', 'dementia']):
            list_intervals.append(rows_subj.iloc[i+1]['age'] - rows_subj.iloc[i]['age'])
mean = np.mean(list_intervals)
std = np.std(list_intervals)
print(f'Interval: {mean} ± {std} years')


# Draw raincloud plot
data = {'interval': list_intervals}
fig, ax = plt.subplots(1, 1, figsize=(2.5, 4))
ax = sns.violinplot(data = data, y = 'interval',
    color='tab:blue', cut=0, width=0.75, inner=None,
    ax=ax, saturation=1, linewidth=1, native_scale=True)

for item in ax.collections:
    x0, y0, width, height = item.get_paths()[0].get_extents().bounds
    item.set_clip_path(plt.Rectangle((x0, y0), width/2, height, transform=ax.transData))
    
num_items = len(ax.collections)
ax = sns.stripplot(data = data, y= 'interval',
    color='tab:blue', jitter=0.15, alpha=0.5, size=2, 
    ax=ax, legend=False, native_scale=True)

for item in ax.collections[num_items:]:
    item.set_offsets(item.get_offsets() + (0.2,0))

ax = sns.boxplot(data = data, y='interval',
    width=0.2, linecolor='black', showfliers=False,
    boxprops=dict(facecolor=(0,0,0,0),
                  linewidth=1, zorder=2),
    whiskerprops=dict(linewidth=1),
    capprops=dict(linewidth=1),
    medianprops=dict(linewidth=1.5, color='tab:orange'),
    ax=ax, native_scale=True)

ax.text(0.02, 0.98, f'interval: {mean:.1f} ± {std:.1f} years',
    fontsize=9, fontfamily='DejaVu Sans',
    transform=ax.transAxes, verticalalignment='top')

ax.tick_params(labelsize=9, which='both')
ax.set_ylabel('interval between CN* and MCI/AD sessions (years)', fontsize=9, fontfamily='DejaVu Sans')

fig.savefig('experiments/2024-03-20_Matched_Cohort_Linear_Model/figs/raincloud_distribution_cn_star_interval_v2.png', dpi=300, bbox_inches='tight')