# Summarize, visualize the distribution of the data in the data bank.
# 
# Author: Chenyu Gao
# Date: Dec 6, 2023

import pandas as pd
import seaborn as sns
from braid.utls import summarize_dataset

MIN = 46
MAX = 92

df = pd.read_csv('/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/quality_assurance/databank_dti_after_pngqa_after_adspqa.csv')

df_base = df.copy()
df_base['dataset_subject'] = df_base['dataset'] + '_' + df_base['subject']
df_base = df_base.groupby('dataset_subject').apply(lambda x: x[x['age'].notnull()].nsmallest(1, 'age')).reset_index(drop=True)

# Distribution plot - scans
plot = sns.displot(df, x="age", col="control_label", hue='dataset', multiple='stack',
                   binwidth=1, height=3, facet_kws=dict(margin_titles=True))

plot.refline(x=MIN, color='red', linestyle='--')
plot.refline(x=MAX, color='red', linestyle='--')

plot.fig.suptitle("Imaging Ages of All Scans from All Subjects", y=1.02)
plot.savefig("./reports/figures/2023-12-06_databank_distribution/age_distribution_all_scans.png", dpi=300)

# Distribution plot - base visit only
plot = sns.displot(df_base, x="age", col="control_label", hue='dataset', multiple='stack',
                   binwidth=1, height=3, facet_kws=dict(margin_titles=True))

plot.refline(x=MIN, color='red', linestyle='--')
plot.refline(x=MAX, color='red', linestyle='--')

plot.fig.suptitle("Imaging Ages of Baseline Scans from All Subjects", y=1.02)
plot.savefig("./reports/figures/2023-12-06_databank_distribution/age_distribution_baseline.png", dpi=300)

# Summary of the dataset
print('------------Overall------------')
df_s = df.loc[(df['age']>=MIN) & (df['age']<=MAX),].copy()
summarize_dataset(df_s)

for dataset in df_s['dataset'].unique():
    print(f"------------{dataset}------------")
    summarize_dataset(df_s.loc[df_s['dataset'] == dataset, ].copy())
