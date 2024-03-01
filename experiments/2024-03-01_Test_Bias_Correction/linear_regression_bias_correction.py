# Test code for bias correction experiment

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Model prediction csv
csv_trainval = 'models/2024-02-07_ResNet101_BRAID_warp/predictions/predicted_age_fold-1_trainval.csv'
csv_test = csv_trainval.replace('trainval', 'test')
df_train = pd.read_csv(csv_trainval)
df_test = pd.read_csv(csv_test)

# Prepare dataframe for linear regression on cross-sectional data
subj_val = np.load('/tmp/.GoneAfterReboot/cross_validation/subjects_fold_1_val.npy', allow_pickle=True)  # train/val splitting
df_train = df_train.loc[df_train['dataset_subject'].isin(subj_val)&
                        (df_train['age_gt']>=45)&(df_train['age_gt']<90), ]  # compute bias correction terms using val set
df_train = df_train.groupby('dataset_subject').apply(lambda x: x.loc[x['age_gt'].idxmin()]).reset_index(drop=True)  # cross-sectional
df_train['BAG'] = df_train['age_pred'] - df_train['age_gt']

# Linear regression
x = sm.add_constant(df_train['age_gt'])
model = sm.OLS(df_train['BAG'], x)
results = model.fit()
slope = results.params['age_gt']
intercept = results.params['const']

# Bias correction
df_train_bc = df_train.copy()
df_train_bc['age_pred'] = df_train['age_pred'] - (slope*df_train['age_gt'] + intercept)
df_train_bc['BAG'] = df_train_bc['age_pred'] - df_train_bc['age_gt']

df_test_bc = df_test.copy()
df_test_bc['age_pred'] = df_test['age_pred'] - (slope*df_test['age_gt'] + intercept)
df_test_bc['BAG'] = df_test_bc['age_pred'] - df_test_bc['age_gt']

df_test['BAG'] = df_test['age_pred'] - df_test['age_gt']

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
sns.scatterplot(x='age_gt', y='BAG', data=df_train, label='Train_before', ax=axes[0, 0])
sns.scatterplot(x='age_gt', y='BAG', data=df_train_bc, label='Train_corrected', ax=axes[1, 0])
sns.scatterplot(x='age_gt', y='BAG', data=df_test, label='Test_before', ax=axes[0, 1])
sns.scatterplot(x='age_gt', y='BAG', data=df_test_bc, label='Test_corrected', ax=axes[1, 1])

for ax in axes.flatten():
    ax.axhline(0, color='black', linestyle='--')
    ax.axvline(45, color='black', linestyle='--')
    ax.axvline(90, color='black', linestyle='--')
    ax.set_xlabel('Chronological Age (years)')
    ax.set_ylabel('Predicted - Chronological (years)')
plt.show()
