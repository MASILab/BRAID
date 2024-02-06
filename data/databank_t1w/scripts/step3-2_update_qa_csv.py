""" After eyeballing the screenshots and removing bad samples,
update the qa.csv to show which samples have been rejected.
run on local machine
"""

import pandas as pd
from pathlib import Path

def update_qa_csv(qa_root, qa_csv, updated_qa_csv):
    df = pd.read_csv(qa_csv)

    for i, row in df.iterrows():
        t1w = row['t1w'].split('/')[-1]
        dataset = t1w.split('_')[0]
        png = Path(qa_root) / dataset / t1w.replace('.nii.gz', '.png')
        
        if not png.is_file():
            df.at[i, 'qa'] = 'rejected'
    
    df.to_csv(updated_qa_csv, index=False)    


if __name__ == '__main__':
    
    update_qa_csv(
        qa_root='data/databank_t1w/quality_assurance/2024-02-05_brain_affine',
        qa_csv='data/databank_t1w/quality_assurance/2024-02-05_brain_affine/qa.csv',
        updated_qa_csv='data/databank_t1w/quality_assurance/2024-02-05_brain_affine/qa_rater1.csv',
    )