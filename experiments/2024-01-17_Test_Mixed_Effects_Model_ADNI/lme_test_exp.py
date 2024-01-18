"""
In this experiment, we test the linear mixed-effects model (LME) implemented in statsmodels.
We use the standard errors of the fixed effects to compute the confidence interval of the regression line.
"""
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import summary_table

covariate = 'age_pred'
outcome = 'memory_score'

# prepare data
df = pd.read_csv('experiments/2024-01-17_Test_Mixed_Effects_Model_ADNI/test_prediction_adni_w_coginfo.csv')
df = df.loc[(df['dataset']=='ADNI') &
            (df['age_gt'] >= 45) & (df['age_gt'] < 90), ]
df = df[['subject', 'session', 'scan', covariate, outcome]]
df = df.groupby(['subject', 'session']).mean().reset_index()

data = df[['subject', 'session', covariate, outcome]].copy()
data.dropna(subset=[outcome], inplace=True)

# fit linear mixed-effects model
model = smf.mixedlm(f"memory_score ~ {covariate}", data=data, groups=data["subject"])
results = model.fit(method=["lbfgs"])

# 95% confidence interval
se_intercept, se_slope = results.bse_fe['Intercept'], results.bse_fe[covariate]
coef_intercept, coef_slope = results.fe_params['Intercept'], results.fe_params[covariate]

x = np.sort(data[covariate].values)
y_reg_mean = x*coef_slope + coef_intercept
y_reg_mean_se = np.sqrt(np.square(se_intercept) + np.square((x - np.mean(x))*se_slope))

ci_upper = y_reg_mean + 1.96*y_reg_mean_se
ci_lower = y_reg_mean - 1.96*y_reg_mean_se

# visualize
fig, ax = plt.subplots()
ax.plot(x, y_reg_mean, color='tab:orange')
ax.fill_between(x, ci_upper, ci_lower, color='tab:orange', alpha=0.2)
ax.scatter(data[covariate], data[outcome], color='tab:blue', s=3)
ax.set_xlabel(covariate)
ax.set_ylabel(outcome)
fig.savefig('experiments/2024-01-17_Test_Mixed_Effects_Model_ADNI/figs/lme_test_exp.png')