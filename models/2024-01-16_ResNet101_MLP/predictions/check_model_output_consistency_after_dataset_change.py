""" The new predictions on the testing set are made on the same dataset, 
except that images were cropped and downsampled before dataloader, rather than within the dataloader.
We want to check how different are the outputs due to this change.
"""

import pandas as pd
from pathlib import Path

pred_dir = Path('models/2024-01-16_ResNet101_MLP/predictions')

for i in range(5):
    fold_idx = i+1
    print(f"-------------------\nFold {fold_idx}")
    
    csv_pre = pred_dir / f"predicted_age_fold-{fold_idx}.csv"
    csv_now = pred_dir / f"predicted_age_fold-{fold_idx}_test.csv"
    
    df_pre = pd.read_csv(csv_pre)
    df_now = pd.read_csv(csv_now)
        
    condition = df_pre['age_pred'] - df_now['age_pred'] > 0.5
    print("Example predictions:")
    print(f"previous:\n{df_pre.loc[condition, 'age_pred'].values}")
    print(f"current:\n{df_now.loc[condition, 'age_pred'].values}")
    print(f"Mean absolute difference: {(df_pre['age_pred'] - df_now['age_pred']).abs().mean()}")
    
    mae_pre = (df_pre['age_pred'] - df_pre['age_gt']).abs().mean()
    mae_now = (df_now['age_pred'] - df_now['age_gt']).abs().mean()

    print(f"\nMAE previous: {mae_pre:.5f}, current: {mae_now:.5f}")
    print()