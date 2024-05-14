import textwrap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

# Data (originally generated in experiments/2024-04-17_MCI_AD_Prediction_From_T_Minus_N_Years_Sliding_Window)
df = pd.read_csv('reports/figures/2024-05-13_MCI_Prediction_From_T_Minus_N_Years_Two_Methods_Merged/data/one_model/data_prep_w-bc.csv')
df_matched = pd.read_csv('reports/figures/2024-05-13_MCI_Prediction_From_T_Minus_N_Years_Two_Methods_Merged/data/one_model/matched_dataset_w-bc_MCI_age-0-1000_hungry_but_picky_match-wo-dataset.csv')
png = 'reports/figures/2024-05-13_MCI_Prediction_From_T_Minus_N_Years_Two_Methods_Merged/figs/figure_data_matching.png'

# Hyperparameters
fontsize = 9
fontfamily = 'DejaVu Sans'
fig = plt.figure(figsize=(6.5, 6), tight_layout=True)
gs = gridspec.GridSpec(nrows=1, ncols=5, wspace=0, hspace=0, width_ratios=[0.325, 1.95, 0.325, 1.95, 1.95])

# subplot-1: y axis label
ax = fig.add_subplot(gs[0])
ax.text(
    0.5, 0.5, f'CN participants (N={df.loc[df["cn_label"]>=0.5, "subj"].nunique()}, sorted by baseline age)', fontsize=fontsize, fontfamily=fontfamily, 
    ha='center', va='center', rotation='vertical', transform=ax.transAxes
    )
ax.axis('off')

# subplot-2: chronological age of CN subjects
ax = fig.add_subplot(gs[1])
data = df.loc[df['cn_label']>=0.5, ].copy()
data = data.sort_values(by='age')
data['y_subject'] = None
for i, subj in enumerate(data['subj'].unique()):
    data.loc[data['subj']==subj, 'y_subject'] = i
sns.lineplot(
    data=data, x='age', y='y_subject', units='subj', estimator=None, 
    lw=0.5, color='tab:gray', alpha=0.5, linestyle='-', ax=ax
    )
cn_groups = {
    0.5: {'name': 'CN (without follow-up sessions)', 'color': 'greenyellow'},
    1: {'name': 'CN (with ≥1 follow-up CN session)', 'color': 'tab:green'},
    }
for cn_label, group_props in cn_groups.items():
    ax.scatter(
        x=data.loc[data['cn_label']==cn_label, 'age'], y=data.loc[data['cn_label']==cn_label, 'y_subject'],
        s=1, c=group_props['color'], label=textwrap.fill(group_props['name'], width=15)
    )
ax.legend(loc='upper left', bbox_to_anchor=(-.01, 1), prop={'size':fontsize, 'family':fontfamily}, labelspacing=1, handlelength=1, handletextpad=0.5)
ax.set_ylabel('')
ax.set_xlabel('chronological age (years)', fontsize=fontsize, fontfamily=fontfamily)
ax.set_ylim(-5, data['subj'].nunique()+5)
ax.set_yticks([])
ax.set_xlim(data['age'].min()-2, data['age'].max()+2)
ax.set_xticks([20, 40, 60, 80, 100])
ax.tick_params(axis='x', which='both', direction='out', length=4, pad=2, labelsize=fontsize, labelfontfamily=fontfamily)

# subplot-3: y axis label
ax = fig.add_subplot(gs[2])
ax.text(
    0.5, 0.5, f'Participants who converted to MCI from CN (N={df.loc[df["age_MCI"].notna(), "subj"].nunique()}) & matched CN data', fontsize=fontsize, fontfamily=fontfamily, 
    ha='center', va='center', rotation='vertical', transform=ax.transAxes
    )
ax.axis('off')

# subplot-4: chronological age of subjects who progressed to MCI/AD (with/without matched CN data points)
ax = fig.add_subplot(gs[3])
data = df.copy()
data = data.sort_values(by='age')
data['y_subject'] = None
for i, subj in enumerate(data.loc[data['time_to_MCI'].notna(), 'subj'].unique()):
    match_ids = df_matched.loc[df_matched['subj']==subj, 'match_id'].unique()
    for _, row in df_matched.loc[df_matched['match_id'].isin(match_ids), ].iterrows():
        data.loc[(data['subj']==row['subj'])&(data['age']==row['age']), 'y_subject'] = i + 0.3
    data.loc[data['subj']==subj, 'y_subject'] = i
data = data.loc[data['y_subject'].notna(), ].copy()

sns.lineplot(
    data=data, x='age', y='y_subject', units='subj', estimator=None,
    lw=0.5, color='tab:gray', alpha=0.5, linestyle='-', ax=ax
)
progression_phases = {
    'normal': {'name': 'CN (at present)', 'color': 'gold'},
    'MCI': {'name': 'MCI', 'color': 'tab:orange'},
    'AD': {'name': 'AD', 'color': 'tab:red'},
}
for phase, props in progression_phases.items():
    ax.scatter(
        x=data.loc[data['time_to_MCI'].notna()&(data['diagnosis']==phase), 'age'], 
        y=data.loc[data['time_to_MCI'].notna()&(data['diagnosis']==phase), 'y_subject'],
        s=1, c=props['color'], label=textwrap.fill(props['name'], width=15)
    )
ax.legend(loc='upper left', bbox_to_anchor=(-.01, 1), prop={'size':fontsize, 'family':fontfamily}, labelspacing=0.5, handlelength=1, handletextpad=0.5)
ax.set_ylabel('')
ax.set_xlabel('chronological age (years)', fontsize=fontsize, fontfamily=fontfamily)
ax.set_ylim(-1, df.loc[df["age_MCI"].notna(), "subj"].nunique()+1)
ax.set_yticks([])
ax.set_xlim(61, 104)
ax.set_xticks([65, 75, 85, 95])
ax.tick_params(axis='x', which='both', direction='out', length=4, pad=2, labelsize=fontsize, labelfontfamily=fontfamily)

for cn_label, group_props in cn_groups.items():
    ax.scatter(
        x=data.loc[data['cn_label']==cn_label, 'age'], y=data.loc[data['cn_label']==cn_label, 'y_subject'],
        s=1, c=group_props['color']
    )

# subplot-5: time to MCI of subjects who progressed to MCI
ax = fig.add_subplot(gs[4])
data = df.copy()
data = data.sort_values(by='age')
data['y_subject'] = None
data_matched = df_matched.copy()
data_matched['y_subject'] = None
for i, subj in enumerate(data.loc[data['time_to_MCI'].notna(), 'subj'].unique()):
    data_matched.loc[data_matched['subj']==subj, 'y_subject'] = i
    data.loc[data['subj']==subj, 'y_subject'] = i
data = data.loc[data['y_subject'].notna(), ].copy()
data_matched = data_matched.loc[data_matched['y_subject'].notna(), ].copy()

sns.lineplot(
    data=data, x='time_to_MCI', y='y_subject', units='subj', estimator=None,
    lw=0.5, color='tab:gray', alpha=0.5, linestyle='-', ax=ax
)
for phase, props in progression_phases.items():
    ax.scatter(
        x=data.loc[data['time_to_MCI'].notna()&(data['diagnosis']==phase), 'time_to_MCI'], 
        y=data.loc[data['time_to_MCI'].notna()&(data['diagnosis']==phase), 'y_subject'],
        s=1, c=props['color']
    )
ax.scatter(
    x=data_matched.loc[data_matched['time_to_MCI'].notna(), 'time_to_MCI'], 
    y=data_matched.loc[data_matched['time_to_MCI'].notna(), 'y_subject'],
    s=8, linewidths=0.5, marker='|', c='k', label=textwrap.fill('with matched CN data points', width=15)
)
ax.legend(loc='upper left', bbox_to_anchor=(-.01, 1), prop={'size':fontsize, 'family':fontfamily}, labelspacing=0.5, handlelength=1, handletextpad=0.5)
ax.set_ylabel('')
ax.set_xlabel('time to MCI (years)', fontsize=fontsize, fontfamily=fontfamily)
ax.set_ylim(-1, df.loc[df["age_MCI"].notna(), "subj"].nunique()+1)
ax.set_yticks([])
ax.set_xlim(-6, 14)
ax.set_xticks([-5, 0, 5, 10])
ax.tick_params(axis='x', which='both', direction='out', length=4, pad=2, labelsize=fontsize, labelfontfamily=fontfamily)

ax.invert_xaxis()
fig.savefig(png, dpi=600)