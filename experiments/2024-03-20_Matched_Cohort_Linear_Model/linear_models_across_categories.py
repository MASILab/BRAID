""" We have Y (WMage - GMage) from four categories, CN, CN*, MCI, and AD.
After matching the data, we perform linear regression in the following form:
    Y = β0 + β1*category + β2*age + β3*sex + β4*category_x_age + ε
"""

import pdb
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('experiments/2024-03-20_Matched_Cohort_Linear_Model/data_matched_cohort.csv')

categories = ['CN', 'CN*', 'MCI', 'AD']

for i, cat_0 in enumerate(categories):
    for j, cat_1 in enumerate(categories[i+1:]):
        
        data = df.loc[df['category_criteria_1'].isin([cat_0, cat_1]), ].copy()
        data['sex'] = data['sex'].map({'female': 0, 'male': 1})
        data['category'] = data['category_criteria_1'].map({cat_0: 0, cat_1: 1})
        data['interact_category_age'] = data['category'] * data['age']
        Y = data['wm_gm_diff']
        X = data[['category', 'age', 'sex', 'interact_category_age']]
        X = sm.add_constant(X)
        model = sm.OLS(Y,X)
        results = model.fit()
        print(f"================================{cat_0} and {cat_1}================================")
        for covar in results.params.keys():
            print(f"{covar.rjust(25)}: \t beta = {results.params[covar]:.3f} \t p-value = {results.pvalues[covar]:.3f}")
        print("===============================================================================\n")
