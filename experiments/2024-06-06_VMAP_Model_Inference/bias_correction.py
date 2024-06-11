import json
import pandas as pd
from pathlib import Path

# # DeepBrainNet
# csv = 'experiments/2024-06-06_VMAP_Model_Inference/results/DeepBrainNet_predictions/vmap_brain_age_deepbrainnet.csv'
# df = pd.read_csv(csv)
# with open('models/2024-04-04_DeepBrainNet/predictions/bias_cor_params.json', 'r') as f:
#     params = json.load(f)
# df['age_pred_dbn_bc'] = df['age_pred_dbn'] - (params['slope']*df['age_chronological'] + params['intercept'])
# df.to_csv(csv.replace('.csv', '_bc.csv'), index=False)

# WMage affine and WMage nonrigid
for transformation in ['affine', 'nonrigid']:
    for fold_idx in [1,2,3,4,5]:
        csv = f'experiments/2024-06-06_VMAP_Model_Inference/results/WMage_{transformation}_predictions/predicted_age_fold-{fold_idx}_WMage_{transformation}.csv'
        df = pd.read_csv(csv)
        if transformation == 'affine':
            js = f'models/2023-12-22_ResNet101/predictions/bias_cor_params_fold-{fold_idx}.json'
        else:
            js = f'models/2024-02-07_ResNet101_BRAID_warp/predictions/bias_cor_params_fold-{fold_idx}.json'
        with open(js, 'r') as f:
            params = json.load(f)
        df['age_pred_bc'] = df['age_pred'] - (params['slope']*df['age'] + params['intercept'])
        df.to_csv(csv.replace('.csv', '_bc.csv'), index=False)
