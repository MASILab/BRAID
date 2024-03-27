# Perform Wilcoxon signed-rank tests to compare the WM and GM age predictions on 
# the same data. We have four categories of data (all matched by age and sex).
#
# "The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples 
# come from the same distribution. In particular, it tests whether the distribution of 
# the differences x - y is symmetric about zero. It is a non-parametric version 
# of the paired T-test." - scipy
# 
# Out of curiosity, we also perform the Wilcoxon rank-sum test to see the results 
# without using the pair assumption.

import pandas as pd
from scipy.stats import wilcoxon, ranksums

df = pd.read_csv('experiments/2024-03-20_Matched_Cohort_Linear_Model/data_matched_cohort.csv')

for cat in df['category_criteria_1'].unique():
    data = df[df['category_criteria_1'] == cat]
    
    # Exclude outliers using IQR
    q1 = data['wm_gm_diff'].quantile(0.25)
    q3 = data['wm_gm_diff'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data = data.loc[(data['wm_gm_diff'] >= lower_bound) & (data['wm_gm_diff'] <= upper_bound)]
    
    print("--------------------Wilcoxon signed-rank test--------------------")
    res = wilcoxon(x=data['wm_gm_diff'])
    print(f"Category: {cat}\tstatistic: {res.statistic}\tp-value: {res.pvalue}")
    
    print("--------------------Wilcoxon rank-sum test--------------------")
    z_stat, p_val = ranksums(data['age_pred_mean_wm'], data['age_pred_mean_gm'])
    print(f"Category: {cat}\tZ-statistic: {z_stat}\tp-value: {p_val}")
    print()