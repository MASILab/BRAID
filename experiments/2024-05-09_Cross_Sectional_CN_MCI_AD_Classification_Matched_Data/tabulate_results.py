import pandas as pd
from pathlib import Path

data_root = Path('experiments/2024-05-09_Cross_Sectional_CN_MCI_AD_Classification_Matched_Data/data')

for suffix in ['_AD_vs_CN.csv', '_CN_vs_MCI.csv', '_CN_vs_CN*.csv']:
    # results of the experiments
    csv_pred = data_root / f'predictions{suffix}'
    csv_report = data_root / f'report{suffix}'
    assert csv_pred.exists() and csv_report.exists(), 'Results not found or named differently.'

    df_pred = pd.read_csv(csv_pred)
    num_pairs = df_pred['match_id'].nunique()

    df_report = pd.read_csv(csv_report)
    classifiers = df_report['clf_name'].unique()
    features = df_report['feat_combo_name'].unique()

    # Format the results in a nice table
    first_col_name = f'features for{suffix.replace(".csv","").replace("_"," ")} ({num_pairs} matched pairs)'
    table = {first_col_name: []}
    for clf in classifiers:
        table[f'{clf} accuracy'] = []
        table[f'{clf} AUC'] = []
    
    # tabulate row by row
    for feat in features:
        table[first_col_name].append(feat)
        for clf in classifiers:
            loc_filter = (df_report['feat_combo_name']==feat) & (df_report['clf_name']==clf)

            acc_mean = df_report.loc[loc_filter, 'acc_mean'].item()
            acc_lower = df_report.loc[loc_filter, 'acc_lower'].item()
            acc_upper = df_report.loc[loc_filter, 'acc_upper'].item()
            table[f'{clf} accuracy'].append(f'{acc_mean:.2f} ({acc_lower:.2f}, {acc_upper:.2f})')

            auc_mean = df_report.loc[loc_filter, 'auc_mean'].item()
            auc_lower = df_report.loc[loc_filter, 'auc_lower'].item()
            auc_upper = df_report.loc[loc_filter, 'auc_upper'].item()
            table[f'{clf} AUC'].append(f'{auc_mean:.2f} ({auc_lower:.2f}, {auc_upper:.2f})')
    
    df_table = pd.DataFrame(table)
    csv_table = data_root / f'table{suffix}'
    df_table.to_csv(csv_table, index=False)
