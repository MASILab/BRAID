import pandas as pd
from tqdm import tqdm

databank_dti = pd.read_csv('/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv')

for dataset in ['train', 'test']:
    braid = pd.read_csv(f'/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/braid_{dataset}.csv')
    braid['diagnosis'] = None
    for i, row in tqdm(braid.iterrows(), total=len(braid.index), desc='Retrieve diagnosis information'):
        loc_filter = (databank_dti['dataset']==row['dataset']) & (databank_dti['subject']==row['subject']) & ((databank_dti['session']==row['session']) | databank_dti['session'].isnull())
        if row['dataset'] in ['UKBB']:
            control_label = databank_dti.loc[loc_filter, 'control_label'].values[0]
            braid.loc[i,'diagnosis'] = 'normal' if control_label == 1 else None
        else:
            braid.loc[i,'diagnosis'] = databank_dti.loc[loc_filter, 'diagnosis_simple'].values[0]
    braid['diagnosis'] = braid['diagnosis'].replace('dementia', 'AD') 
    braid = braid.loc[braid['diagnosis'].isin(['normal', 'MCI', 'AD']), ]
    num_subj = braid['dataset_subject'].nunique()
    print(f"Number of subjects in BRAID {dataset} set: {num_subj}")