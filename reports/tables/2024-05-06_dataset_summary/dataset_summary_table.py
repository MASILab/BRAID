import pandas as pd
from tqdm import tqdm
from tabulate import tabulate

def report_dataset_summary(df):
    header = ['study', 'n participants (male/female)', 'n participants control and disease',
              'n sessions control and disease', 'n scans', 'age mean ± std', 'age range']
    table_data = []
    
    df['subj'] = df['dataset'] + '_' + df['subject']
    df['ses'] = df['subj'] + '_' + df['session']
    
    # subset for each row
    dict_subsets = {dataset: df.loc[df['dataset']==dataset, ] for dataset in sorted(df['dataset'].unique())}
    dict_subsets['Combined'] = df
    
    for subset_name, subset in dict_subsets.items():
        n_subj = subset['subj'].nunique()
        n_subj_male = subset.loc[subset['sex']=='male', 'subj'].nunique()
        n_subj_female = subset.loc[subset['sex']=='female', 'subj'].nunique()
        
        subj_ad = subset.loc[subset['diagnosis']=='AD', 'subj'].unique()
        n_subj_ad = len(subj_ad)
        subj_mci = subset.loc[(subset['diagnosis']=='MCI')&(~subset['subj'].isin(subj_ad)), 'subj'].unique()
        n_subj_mci = len(subj_mci)
        n_subj_cn = subset.loc[(subset['diagnosis']=='normal')&(~subset['subj'].isin(subj_ad))&(~subset['subj'].isin(subj_mci)), 'subj'].nunique()
        
        n_ses = subset['ses'].nunique()
        n_ses_cn = subset.loc[subset['diagnosis']=='normal', 'ses'].nunique()
        n_ses_mci = subset.loc[subset['diagnosis']=='MCI', 'ses'].nunique()
        n_ses_ad = subset.loc[subset['diagnosis']=='AD', 'ses'].nunique()
        
        n_scans = subset.shape[0]
        
        age_mean = subset['age'].mean()
        age_std = subset['age'].std()
        age_min = subset['age'].min()
        age_max = subset['age'].max()
        
        table_data.append([
            subset_name,
            f'{n_subj} ({n_subj_male}/{n_subj_female})',
            f'{n_subj_cn} CN, {n_subj_mci} MCI, {n_subj_ad} AD',
            f'{n_ses} ({n_ses_cn} CN, {n_ses_mci} MCI, {n_ses_ad} AD)',
            n_scans,
            f'{age_mean:.1f} ± {age_std:.1f}',
            f'{age_min:.1f} - {age_max:.1f}',
        ])
    
    print(tabulate(table_data, header, tablefmt='grid'))

if __name__ == '__main__':
    databank_dti = pd.read_csv('/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv')
    qa = pd.read_csv('/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/quality_assurance/databank_dti_v1_after_pngqa_after_adspqa.csv')
    qa['diagnosis'] = None
    
    for i, row in tqdm(qa.iterrows(), total=len(qa.index), desc='Retrieve diagnosis information'):
        loc_filter = (databank_dti['dataset']==row['dataset']) & (databank_dti['subject']==row['subject']) & ((databank_dti['session']==row['session']) | databank_dti['session'].isnull())
        if row['dataset'] in ['UKBB']:
            control_label = databank_dti.loc[loc_filter, 'control_label'].values[0]
            qa.loc[i,'diagnosis'] = 'normal' if control_label == 1 else None
        else:
            qa.loc[i,'diagnosis'] = databank_dti.loc[loc_filter, 'diagnosis_simple'].values[0]
    qa['diagnosis'] = qa['diagnosis'].replace('dementia', 'AD') 
    qa = qa.loc[qa['diagnosis'].isin(['normal', 'MCI', 'AD']), ]
    
    report_dataset_summary(qa)
    