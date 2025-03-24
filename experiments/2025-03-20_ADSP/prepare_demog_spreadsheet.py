import pdb
import pandas as pd
from pathlib import Path

# csv file name: (dataset name in BIDS, is cross-sectional)
manually_checked = {
    'ADNI_info.csv': ('ADNI_DTI', False),
    'ICBM_info.csv': ('ICBM', True),
    'BIOCARD_info.csv': ('BIOCARD', False),
}

lut_sex = {
    'female': 0,
    'male': 1, 
}

lut_race = {
    'White': 1, 
    'Asian': 2, 
    'Black or African American': 3, 
    'American Indian or Alaska Native': 4,
    'More than one race': 0, 
    'Native Hawaiian or Other Pacific Islander': 0,
    'Hispanic or Latino': 0,
}

path_bids = Path('/nfs2/harmonization/BIDS')
raw_dir = Path('experiments/2025-03-20_ADSP/raw')
converted_dir = Path('experiments/2025-03-20_ADSP/converted')

for fn_csv in manually_checked.keys():
    print(f"Converting {fn_csv}")
    df = pd.read_csv(raw_dir / fn_csv)
    print(f"\tNumber of rows (before): {len(df.index)}")
    
    # Are all unique labels considered in the lookup tables
    for label in df['sex'].unique():
        if label not in lut_sex and not pd.isna(label):
            raise ValueError(f"sex label not considered: {label}")        
    for label in df['race'].unique():
        if label not in lut_race and not pd.isna(label):
            raise ValueError(f"race label not considered: {label}")
        
    df['sex'] = df['sex'].apply(lambda x: lut_sex.get(x, None) if not pd.isna(x) else None)
    df['race'] = df['race'].apply(lambda x: lut_race.get(x, None) if not pd.isna(x) else None)
    
    # Make sure every session in BIDS has a row, even if the demographic information is missing
    path_dataset = path_bids / manually_checked[fn_csv][0]
    for subject in path_dataset.glob('sub-*'):
        if manually_checked[fn_csv][1]:
            for session in subject.glob('ses-*'):
                if len(df.loc[(df['subject']==subject.name)&(df['session']==session.name), ].index) == 0:
                    new_row = {col: None for col in df.columns}
                    new_row['subject'] = subject.name
                    new_row['session'] = session.name
                    df = df.append(new_row, ignore_index=True)
        else:
            if len(df.loc[df['subject']==subject.name,].index) == 0:
                new_row = {col: None for col in df.columns}
                new_row['subject'] = subject.name
                df = df.append(new_row, ignore_index=True)
                
    new_csv = converted_dir / fn_csv.replace('.csv', '_converted.csv')
    df.to_csv(new_csv, index=False)
    print(f"\tNumber of rows (after): {len(df.index)}")