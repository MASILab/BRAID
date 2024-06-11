import pandas as pd
import subprocess
from pathlib import Path

Path('experiments/2024-06-06_VMAP_Model_Inference/results/delivery/2024-06-10_VMAP_BrainAges_delivery/').mkdir(parents=True, exist_ok=True)

# DeepBrainNet
subprocess.run([
    'rsync', '-L', 'experiments/2024-06-06_VMAP_Model_Inference/results/DeepBrainNet_predictions/vmap_brain_age_deepbrainnet_bc.csv',
    'experiments/2024-06-06_VMAP_Model_Inference/results/delivery/2024-06-10_VMAP_BrainAges_delivery/vmap_brain_age_gm_age_deepbrainnet.csv',
    ])  # simply rename

# BRAID
for transformation in ['affine', 'nonrigid']:
    for fold_idx in [1,2,3,4,5]:
        csv = f'experiments/2024-06-06_VMAP_Model_Inference/results/WMage_{transformation}_predictions/predicted_age_fold-{fold_idx}_WMage_{transformation}_bc.csv'
        csv_new = f'experiments/2024-06-06_VMAP_Model_Inference/results/delivery/2024-06-10_VMAP_BrainAges_delivery/vmap_brain_age_wm_age_{transformation}_fold-{fold_idx}.csv'
        
        df = pd.read_csv(csv)
        df['bids_subject_id'] = df['subject']
        df['bids_session_id'] = df['session']
        df['age_chronological'] = df['age']
        df[f'age_pred_wm_age_{transformation}'] = df['age_pred']
        df[f'age_pred_wm_age_{transformation}_bc'] = df['age_pred_bc']
        
        df_new = df[[
            'bids_subject_id','bids_session_id','bids_acquisition_label','bids_run_label',
            'xnat_project','vmac_id','session_id','age_chronological',
            f'age_pred_wm_age_{transformation}', f'age_pred_wm_age_{transformation}_bc',
            ]].copy()
        df_new.to_csv(csv_new, index=False)