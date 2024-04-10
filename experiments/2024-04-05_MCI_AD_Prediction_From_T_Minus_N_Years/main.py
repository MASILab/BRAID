import pandas as pd
from functions.data import roster_brain_age_models, DataPreparation

d = DataPreparation(
    dict_models = roster_brain_age_models(), 
    databank_csv = '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv',
)
df = d.load_predictions_of_all_models(bias_correction=True)
df = d.retrieve_diagnosis_label(df)
df = d.assign_cn_label(df)
df = d.feature_engineering(df)
df = d.mark_progression_subjects_out(df)
# df.to_csv('experiments/2024-04-05_MCI_AD_Prediction_From_T_Minus_N_Years/data/tmp.csv', index=False)
d.visualize_data_points(df, png='experiments/2024-04-05_MCI_AD_Prediction_From_T_Minus_N_Years/figs/vis_progression_data_points_MCI.png', disease='MCI')
d.visualize_data_points(df, png='experiments/2024-04-05_MCI_AD_Prediction_From_T_Minus_N_Years/figs/vis_progression_data_points_AD.png', disease='AD')

dict_subsets = d.get_preclinical_subsets(df, disease='MCI', method='index cut', num_subsets=11)