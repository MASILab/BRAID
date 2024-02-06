import pandas as pd
from pathlib import Path

def tabulate_QA_results(
    png_root,
    databank_dti_csv,
    qa_csv,
):
    png_root = Path(png_root)
    
    df = pd.read_csv(databank_dti_csv)
    df = df[['dataset', 'subject', 'session', 'scan', 'prequal_folder', 'wmatlas_folder']]
    df['session'] = df['session'].fillna('ses-1')
    df['qa'] = None

    for i, row in df.iterrows():
        png = png_root / row['dataset'] / f"{row['subject']}_{row['session']}_scan-{row['scan']}.png"
        if not png.is_file():
            df.at[i, 'qa'] = 'rejected'
    
    n_total = len(df.index)
    n_rejected = len(df.loc[df['qa'] == 'rejected', ].index)
    
    print(f"After the PNG QA, {n_rejected} out of {n_total} scans are rejected.")
    df.to_csv(qa_csv, index=False)
    
    
if __name__ == '__main__':
    tabulate_QA_results(
        png_root='data/databank_dti/quality_assurance/QA_databank_dti_done',
        databank_dti_csv='/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v1.csv',
        qa_csv='data/databank_dti/quality_assurance/2023-12-05_PNG_QA_affine_registered_skullstripped_FA_MD.csv',
    )