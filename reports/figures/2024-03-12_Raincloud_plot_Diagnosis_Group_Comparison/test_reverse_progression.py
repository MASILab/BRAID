""" Is there any MCI/AD subject that converted back to CN?

MCI -> CN: 3 ADNI subjects
AD -> MCI/CN: 0
"""
import pandas as pd

df = pd.read_csv('reports/figures/2024-03-12_Raincloud_plot_Diagnosis_Group_Comparison/tmp.csv')

df['subj'] = df['dataset'] + '_' + df['subject']

for subj in df['subj'].unique():
    df_subj = df.loc[df['subj'] == subj, ].copy()
    df_subj = df_subj.sort_values(by='age_gt')
    
    for i in range(len(df_subj.index)-1):
        if (df_subj.iloc[i]['diagnosis'] in ['MCI', 'dementia']) & (df_subj.iloc[i+1]['diagnosis']=='normal'):
            print(df_subj)
        
        if (df_subj.iloc[i]['diagnosis'] in ['dementia']) & (df_subj.iloc[i+1]['diagnosis']=='MCI'):
            print(df_subj)