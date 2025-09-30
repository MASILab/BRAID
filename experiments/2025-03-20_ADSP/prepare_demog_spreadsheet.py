import pdb
import json
import argparse
import pandas as pd
from pathlib import Path

lut_sex = {
    'female': 0,
    'male': 1,
    0: 0, 
    1: 1,
    '0': 0,
    '1': 1,
}

lut_race = {
    'NHW': 1,
    'White': 1, 
    'Asian': 2, 
    'NHB': 3,
    'Black': 3,
    'Black or African American': 3, 
    'American Indian or Alaska Native': 4,
    'American Indian or Alaskan Native': 4,
    'More than one race': 0, 
    'Native Hawaiian or Other Pacific Islander': 0,
    'Hispanic or Latino': 0,
    'Other': 0,
    'Other, Unknown, or More than one race': 0,
    'Unknown': 0,
    'Other/unknown/multiple': 0,
}


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


def combine_csvs(
        prev_csv: str = 'raw/ADNI_info.csv',
        new_csv: str = 'raw/ADNI_6_10_25/bids.csv',
        output_csv: str = 'raw/ADNI_info_combined.csv',
    ):
    """ Combine two csv files by concatenation and removing duplicates.
    """
    df_prev = pd.read_csv(prev_csv)
    df_new = pd.read_csv(new_csv)
    df_combined = pd.concat([df_prev, df_new], join='inner', ignore_index=True)
    df_combined.drop_duplicates(keep='first', inplace=True)
    
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description='Prepare demographic spreadsheet for a BIDS dataset to run BRAID.')
    parser.add_argument('-n', '--bids_name', type=str, required=True, help='Name of the dataset in BIDS.')
    parser.add_argument('-c', '--cross_sectional', action='store_true', help='Whether the dataset is cross-sectional.')
    parser.add_argument('-i', '--input_csv', type=str, required=True, help='Path to the csv to be converted.')
    parser.add_argument('-o', '--outdir', type=str, required=False, default='./converted', help='Directory to save the converted csv.')
    args = parser.parse_args()

    path_dataset = Path(f'/nfs2/harmonization/BIDS/{args.bids_name}')
    input_csv = Path(args.input_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_csv = outdir / input_csv.name.replace('.csv', '_converted.csv')
    
    df = pd.read_csv(input_csv)
    for col in ['subject','age','sex','race']:
        assert col in df.columns, f"Column {col} not found in {input_csv}"
    if not args.cross_sectional:
        assert 'session' in df.columns, f"Column session not found in {input_csv}"

    for label in df['sex'].unique():
        if label not in lut_sex and not pd.isna(label):
            raise NotImplementedError(f"Sex label not considered: {label}")
    for label in df['race'].unique():
        if label not in lut_race and not pd.isna(label):
            raise NotImplementedError(f"Race label not considered: {label}")
    
    df['sex'] = df['sex'].apply(lambda x: lut_sex.get(x, None) if not pd.isna(x) else None)
    df['race'] = df['race'].apply(lambda x: lut_race.get(x, None) if not pd.isna(x) else None)
    
    print("Scanning BIDS directory to include missing rows as placeholders...")
    if args.cross_sectional:
        bids_subjects = [{'subject': subject.name} for subject in path_dataset.glob('sub-*')]
        if not bids_subjects:
            raise FileNotFoundError(f"No subjects found in BIDS directory: {path_dataset}")
        
        bids_df = pd.DataFrame(bids_subjects)
        df = pd.merge(bids_df, df, on='subject', how='left')

    else:
        bids_sessions = [
            {'subject': subject.name, 'session': session.name}
            for subject in path_dataset.glob('sub-*')
            for session in subject.glob('ses-*')
        ]
        if not bids_sessions:
            raise FileNotFoundError(f"No sessions found in BIDS directory: {path_dataset}")

        bids_df = pd.DataFrame(bids_sessions)
        df = pd.merge(bids_df, df, on=['subject', 'session'], how='left')

    id_cols = ['subject', 'session'] if not args.cross_sectional else ['subject']
    df = df.drop_duplicates(subset=id_cols, keep='first')
                
    cols_to_keep = ['subject','session','age','sex','race']
    df = df[cols_to_keep]
    df.to_csv(output_csv, index=False)
    print(f"Converted csv saved to {output_csv}")


if __name__ == '__main__':
    # extract_dwi_tabular_adni_6_10_25()
    # combine_csvs()
    
    main()
    