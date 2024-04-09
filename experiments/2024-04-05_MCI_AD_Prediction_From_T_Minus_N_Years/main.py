from functions.data import roster_brain_age_models, DataPreparation

d = DataPreparation(
    dict_models = roster_brain_age_models(), 
    databank_csv = '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv',
)
df = d.load_predictions_of_all_models(bias_correction=True)
df = d.retrieve_diagnosis_label(df)
df = d.mark_progression_subjects_out(df)
d.visualize_data_points(df, png='experiments/2024-04-05_MCI_AD_Prediction_From_T_Minus_N_Years/figs/vis_progression_data_points_MCI.png', disease='MCI')
d.visualize_data_points(df, png='experiments/2024-04-05_MCI_AD_Prediction_From_T_Minus_N_Years/figs/vis_progression_data_points_AD.png', disease='AD')
