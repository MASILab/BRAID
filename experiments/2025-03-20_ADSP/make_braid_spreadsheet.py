import re
import pdb
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# datasets to include in 2025 ADSP delivery
delivery_datasets = [
    "ADNI_DTI",
    "NACC",
    "ROSMAPMARS",
    "WASHU",
    "WRAP",
    "WRAPnew",
    "HABSHD",
    "SCAN",
    "FloridaADRC",
    "Indiana",
]


def _is_dataset_cross_sectional(path_root: str | Path) -> bool:
    path_root = Path(path_root)
    assert path_root.is_dir(), f"{path_root} is not a directory."
    
    example_subject = next(path_root.glob("sub-*"))
    has_session_layer = any(
        p.is_dir() and p.name.startswith('ses-') for p in example_subject.iterdir()
    )
    return not has_session_layer


def tabulate_braid_results(dataset_name, path_output_csv):
    """ collect braid results from one given dataset
    """
    path_derivatives = Path(f"/nfs2/harmonization/BIDS/{dataset_name}/derivatives/")
    cross_sectional = _is_dataset_cross_sectional(path_derivatives)
    
    df = pd.DataFrame()
    search_pattern = "sub-*/BRAID*/final/braid_predictions.csv" if cross_sectional else "sub-*/ses-*/BRAID*/final/braid_predictions.csv"
    all_csvs = list(path_derivatives.glob(search_pattern))
    
    for path_csv in tqdm(all_csvs, desc=f"collect from {dataset_name}"):
        if cross_sectional:
            subject = path_csv.parts[-4]
            session = None
        else:
            subject = path_csv.parts[-5]
            session = path_csv.parts[-4]
        
        braid_fd = path_csv.parts[-3]
        match = re.match(r'^BRAID(?:(acq-\w+))?(?:(run-\d{1,2}))?$', braid_fd)
        acq = match.group(1)
        run = match.group(2)
        
        df_one = pd.read_csv(path_csv)
        df_one['path_csv'] = str(path_csv)
        df_one['dataset'] = dataset_name
        df_one['subject'] = subject
        df_one['session'] = session
        df_one['acq'] = acq
        df_one['run'] = run
        df = pd.concat([df, df_one], ignore_index=True)
    
    df.to_csv(path_output_csv, index=False)
    return df


def make_braid_spreadsheet(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    combined = []
    
    for dataset in delivery_datasets:
        path_output_csv = outdir / f"{dataset}_braid_results.csv"
        if not path_output_csv.exists():
            df = tabulate_braid_results(dataset, path_output_csv)
        else:
            try:
                df = pd.read_csv(path_output_csv)
            except pd.errors.EmptyDataError:
                print(f"Warning: {path_output_csv} is empty. Skipping.")
                continue
        print(f"{dataset}: {len(df)} records collected.")
        combined.append(df)

    df_all = pd.concat(combined, ignore_index=True)
    
    # re-arrange columns
    cols = ['dataset','subject','session','acq','run']
    for c in df_all.columns:
        if c in ['path_dwi','path_t1w','path_csv']:
            continue
        if c not in cols:
            cols.append(c)
    df_all = df_all[cols]
    df_all.to_csv(outdir / "braid_combined.csv", index=False)


if __name__ == "__main__":
    make_braid_spreadsheet(
        outdir='/nfs/masi/gaoc11/projects/BRAID/experiments/2025-03-20_ADSP/delivery_2025-12-12'
        )