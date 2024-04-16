import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from functions.data import prepare_subsets, roster_feature_combinations
from functions.classification import roster_classifiers, evaluate_clf_perf
from functions.visualization import visualize_t_minus_n_prediction_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--disease', type=str, default='MCI')
    parser.add_argument('--num_subsets', type=int, default=7)
    parser.add_argument('--bc', type=bool, default=True, help='whether to use bias corrected predictions or not')
    parser.add_argument('--figdir', type=str, default='experiments/2024-04-05_MCI_AD_Prediction_From_T_Minus_N_Years/figs/')
    parser.add_argument('--outdir', type=str, default='experiments/2024-04-05_MCI_AD_Prediction_From_T_Minus_N_Years/results/')
    parser.add_argument('--databank_csv', type=str, default='/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv')
    args = parser.parse_args()

    dict_subsets, dict_subsets_matched = prepare_subsets(args)
    feat_combo = roster_feature_combinations(dict_subsets[0])
    classifiers = roster_classifiers()

    dict_results = {
        'classifier': [],
        'feature_combination': [],
        'subset_id': [],
        'time_to_event_mean': [],
        'auc_mean': [],
        'auc_std': [],
        }
    
    for classifier_name in classifiers.keys():
        model_class, kwargs = classifiers[classifier_name]
        for combo_name, list_features in feat_combo.items():
            for subset_id in dict_subsets.keys():
                df_subset = dict_subsets[subset_id]
                df_subset_m = dict_subsets_matched[subset_id]
                
                auc_mean, auc_std = evaluate_clf_perf(model_class, kwargs, list_features, df_subset_m)
                
                dict_results['classifier'].append(classifier_name)
                dict_results['feature_combination'].append(combo_name)
                dict_results['subset_id'].append(subset_id)
                dict_results['time_to_event_mean'].append(df_subset[f'time_to_{args.disease}'].mean())
                dict_results['auc_mean'].append(auc_mean)
                dict_results['auc_std'].append(auc_std)

    results = pd.DataFrame(data=dict_results)
    results.to_csv(Path(args.outdir)/f'{args.disease}_{args.num_subsets}_{"bc" if args.bc else "wobc"}.csv', index=False)
    
    visualize_t_minus_n_prediction_results(results, dict_subsets, png=Path(args.figdir)/f'{args.disease}_{args.num_subsets}_{"bc" if args.bc else "wobc"}.png')