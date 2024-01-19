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
pvalue_intercept, pvalue_slope = results.pvalues['Intercept'], results.pvalues[covariate]

x = np.sort(data[covariate].values)
y_reg_mean = x*coef_slope + coef_intercept
y_reg_mean_se = np.sqrt(np.square(se_intercept) + np.square((x - np.mean(x))*se_slope))

ci_upper = y_reg_mean + 1.96*y_reg_mean_se
ci_lower = y_reg_mean - 1.96*y_reg_mean_se

# spaghetti plot + regression line + confidence intervals
figsize = (4, 4)
dpi = 300
fontsize = {'legend': 8, 'label': 10}
fontfamily = 'DejaVu Sans'
color = {'spaghetti': 'tab:blue', 'regression': 'tab:red'}
alpha = {'spaghetti': 0.1, 'regression': 1, 'confidence': 0.2}

fig, ax = plt.subplots(figsize=figsize)

for subj in data['subject'].unique():
    df_plot = data.loc[data['subject']==subj, ].sort_values(by=covariate)
    ax.plot(df_plot[covariate], df_plot[outcome], '.-', color = color['spaghetti'], markersize=3, markeredgewidth=0, linewidth= 1, alpha=alpha['spaghetti'])

ax.plot(x, y_reg_mean, color=color['regression'], label=f'β={coef_slope:.3f} p-value={pvalue_slope:.1e}')
ax.fill_between(x, ci_upper, ci_lower, color=color['regression'], alpha=alpha['confidence'], label='95% confidence interval')
ax.legend(prop={'size': fontsize['legend'], 'family': fontfamily})

ax.set_xlabel(covariate, fontsize=fontsize['label'], fontfamily=fontfamily)
ax.set_ylabel(outcome, fontsize=fontsize['label'], fontfamily=fontfamily)
fig.savefig('experiments/2024-01-17_Test_Mixed_Effects_Model_ADNI/figs/lme_test_exp.png', dpi=dpi)
