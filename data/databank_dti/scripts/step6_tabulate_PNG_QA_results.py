import pandas as pd
from pathlib import Path

def tabulate_QA_results(
    png_root,
    csv
):
    d = {'dataset': [], 'scan_passed_QA': []}
    
    for dataset in Path(png_root).iterdir():
        if not dataset.is_dir():
            continue
        for png in dataset.iterdir():
            d['dataset'].append(dataset.name)
            d['scan_passed_QA'].append(png.name.replace('.png', ''))
    
    df = pd.DataFrame(data = d)
    df.to_csv(csv, index=False)

if __name__ == '__main__':
    tabulate_QA_results(
        png_root = '/nfs/masi/gaoc11/projects/BRAID/data/databank_dti/quality_assurance/QA_databank_dti_done',
        csv = 'data/databank_dti/quality_assurance/2023-12-05_PNG_QA_affine_registered_skullstripped_FA_MD.csv'
    )
