import pdb
from functions.data import DataPreparation, roster_brain_age_models, roster_feature_combinations
from functions.classification import roster_classifiers, run_classification_experiments

# Load dataset
dict_models = roster_brain_age_models()
d = DataPreparation(dict_models, databank_csv='/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv')
df = d.load_predictions_of_all_models()
df = d.retrieve_diagnosis_label(df)
df = d.assign_fine_category_label(df)
df = d.feature_engineering(df)
# d.draw_histograms_columns(df, cols_exclude=['dataset','subject','session','subj'],
                        #   save_dir='experiments/2024-03-28_Cross_Sectional_CN_MCI_AD_Classification_Matched_Data/figs/histograms')

feat_combo = roster_feature_combinations(df)
classifiers = roster_classifiers()
# CN vs. AD classification
data =  d.match_data(df, category_col='category_criteria_2', match_order=['AD', 'CN'])
data = d.split_data_into_k_folds(data, category_col='category_criteria_2')
data['category'] = data['category_criteria_2'].map({'CN': 0, 'AD': 1})
data.to_csv('experiments/2024-03-28_Cross_Sectional_CN_MCI_AD_Classification_Matched_Data/data/data_CN_vs_AD.csv', index=False)
run_classification_experiments(
    data, feat_combo, classifiers, 
    results_csv='experiments/2024-03-28_Cross_Sectional_CN_MCI_AD_Classification_Matched_Data/data/results_CN_vs_AD.csv', 
    num_folds=5, impute_method='mean', feature_selection_method='backward')


# # CN vs. MCI classification
# data =  d.match_data(df, category_col='category_criteria_2', match_order=['MCI', 'CN'])
# data = d.split_data_into_k_folds(data, category_col='category_criteria_2')
# data.to_csv('experiments/2024-03-28_Cross_Sectional_CN_MCI_AD_Classification_Matched_Data/data/data_CN_vs_MCI.csv', index=False)



# # CN vs. CN* classification
# data =  d.match_data(df, category_col='category_criteria_1', match_order=['CN*', 'CN'])
# data = d.split_data_into_k_folds(data, category_col='category_criteria_1')
# data.to_csv('experiments/2024-03-28_Cross_Sectional_CN_MCI_AD_Classification_Matched_Data/data/data_CN_vs_CNstar.csv', index=False)
