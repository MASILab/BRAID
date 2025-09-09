import pdb
import json
import pandas as pd
from pathlib import Path


def extract_dwi_tabular_adni_6_10_25(
    input_dir: str = 'raw/ADNI_6_10_25',
    output_csv: str = 'raw/ADNI_6_10_25/bids.csv',
    ):
    """ For the recently downloaded ADNI DWI data (as of September 5, 2025),
    extract and organize the tabular information (age, sex, race) into a csv.
    """
    input_dir = Path(input_dir)
    required_files = [
        'idaSearch_6_10_2025.csv',
        'PTDEMOG_05Sep2025.csv',
        'race_dict.json',
        'visit_mapping.json',
    ]
    for fn in required_files:
        if not (input_dir / fn).exists():
            raise FileNotFoundError(f"Required file not found: {dir / fn}")
    
    with open(input_dir/'race_dict.json', 'r') as f:
        race_dict = json.load(f)
    with open(input_dir/'visit_mapping.json', 'r') as f:
        visit_mapping = json.load(f)
    
    idasearch = pd.read_csv(input_dir / 'idaSearch_6_10_2025.csv')
    ptdemog = pd.read_csv(input_dir / 'PTDEMOG_05Sep2025.csv')
    ptdemog = ptdemog.loc[ptdemog['PTRACCAT'].isin(race_dict.keys()), ['PTID','PTRACCAT']].copy()
    ptdemog.sort_values(by='PTRACCAT', inplace=True)
    ptdemog.drop_duplicates(subset=['PTID'], keep='first', inplace=True)
    
    df = pd.merge(left=idasearch, right=ptdemog, how='left', left_on='Subject ID', right_on='PTID')
    df['site'] = df['Subject ID'].str.split('_S_').str[0]
    df['subject'] = df['Subject ID'].str.split('_S_').str[1].apply(lambda x: f"sub-{x}")
    df['session'] = df['Visit'].apply(lambda x: visit_mapping[x]).apply(lambda x: f"ses-{x}")
    df['sex'] = df['Sex'].apply(lambda x: 1 if x=='M' else (0 if x=='F' else None))
    df['age'] = df['Age']
    df['diagnosis'] = df['Research Group']
    df['race'] = df['PTRACCAT'].apply(lambda x: race_dict.get(x, None) if not pd.isna(x) else None)
    
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    cols_to_save = ['site','subject','session','age','sex','diagnosis','race']
    df = df[cols_to_save]
    df.to_csv(output_csv, index=False)


if __name__ == '__main__':
    extract_dwi_tabular_adni_6_10_25()


# # csv file name: (dataset name in BIDS, is cross-sectional)
# manually_checked = {
#     # 'ADNI_info.csv': ('ADNI_DTI', False),
#     # 'ICBM_info.csv': ('ICBM', True),
#     # 'BIOCARD_info.csv': ('BIOCARD', False),
#     # 'ROSMAPMARS_info.csv': ('ROSMAPMARS', False),
#     # 'NACC_info.csv': ('NACC', False),
#     'BLSA_info.csv': ('BLSA', False),
#     'OASIS3_info.csv': ('OASIS3', False),
#     'OASIS4_info.csv': ('OASIS4', False),
#     'WRAP_info.csv': ('WRAP', False),
#     'HABSHD_info.csv': ('HABSHD', False),
# }

# lut_sex = {
#     'female': 0,
#     'male': 1,
#     0: 0, 
#     1: 1,
# }

# lut_race = {
#     'White': 1, 
#     'Asian': 2, 
#     'Black or African American': 3, 
#     'American Indian or Alaska Native': 4,
#     'More than one race': 0, 
#     'Native Hawaiian or Other Pacific Islander': 0,
#     'Hispanic or Latino': 0,
#     'Black': 3,
#     'NHW': 1,
#     'Other': 0,
#     'NHB': 3,
#     'Other, Unknown, or More than one race': 0,
# }

# path_bids = Path('/nfs2/harmonization/BIDS')
# raw_dir = Path('experiments/2025-03-20_ADSP/raw')
# converted_dir = Path('experiments/2025-03-20_ADSP/converted')

# for fn_csv in manually_checked.keys():
#     print(f"Converting {fn_csv}")
#     df = pd.read_csv(raw_dir / fn_csv)
#     print(f"\tNumber of rows (before): {len(df.index)}")
    
#     # Are all unique labels considered in the lookup tables
#     for label in df['sex'].unique():
#         if label not in lut_sex and not pd.isna(label):
#             raise ValueError(f"sex label not considered: {label}")        
#     for label in df['race'].unique():
#         if label not in lut_race and not pd.isna(label):
#             raise ValueError(f"race label not considered: {label}")
        
#     df['sex'] = df['sex'].apply(lambda x: lut_sex.get(x, None) if not pd.isna(x) else None)
#     df['race'] = df['race'].apply(lambda x: lut_race.get(x, None) if not pd.isna(x) else None)
    
#     # Make sure every session in BIDS has a row, even if the demographic information is missing
#     path_dataset = path_bids / manually_checked[fn_csv][0]
#     cross_sectional = manually_checked[fn_csv][1]
#     assert 'subject' in df.columns, f"Column subject not found in {fn_csv}"
#     if not cross_sectional:
#         assert 'session' in df.columns, f"Column session not found in {fn_csv}"
        
#     for subject in path_dataset.glob('sub-*'):
#         if not cross_sectional:
#             for session in subject.glob('ses-*'):
#                 if len(df.loc[(df['subject']==subject.name)&(df['session']==session.name), ].index) == 0:
#                     new_row = {col: None for col in df.columns}
#                     new_row['subject'] = subject.name
#                     new_row['session'] = session.name
#                     new_row = pd.DataFrame([new_row])
#                     df = pd.concat([df, new_row], ignore_index=True)
#         else:
#             if len(df.loc[df['subject']==subject.name,].index) == 0:
#                 new_row = {col: None for col in df.columns}
#                 new_row['subject'] = subject.name
#                 new_row = pd.DataFrame([new_row])
#                 df = pd.concat([df, new_row], ignore_index=True)
    
#     # Remove duplicates
#     subset = ['subject', 'session'] if not cross_sectional else ['subject']
#     df = df.drop_duplicates(subset=subset, keep='first')
                
#     new_csv = converted_dir / fn_csv.replace('.csv', '_converted.csv')
#     df.to_csv(new_csv, index=False)
#     print(f"\tNumber of rows (after): {len(df.index)}")