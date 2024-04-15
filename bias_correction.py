import re
import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
from pathlib import Path

class BiasCorrection:
    def __init__(self, prediction_root, crossval_subjects_dir):
        self.prediction_root = Path(prediction_root)
        self.crossval_subjects_dir = Path(crossval_subjects_dir)
    
    
    def get_trainval_test_csv_dict(self):
        
        dict_prediction_csv = {'trainval':[], 'test':[]}
        fn_pattern = 'predicted_age_trainval.csv' if 'DeepBrainNet' in str(self.prediction_root) else 'predicted_age_fold-*_trainval.csv'
        
        for csv_trainval in self.prediction_root.glob(fn_pattern):
            csv_test = csv_trainval.parent / csv_trainval.name.replace('trainval', 'test')
            if csv_test.is_file():
                dict_prediction_csv['trainval'].append(str(csv_trainval))
                dict_prediction_csv['test'].append(str(csv_test))
        
        return dict_prediction_csv
    
    
    def get_subj_val(self, fn):
        """ Get the subject list for validation set according to the file name (string)
        """
        match = re.search(r'fold-(\d+)', fn)
        fold_idx = int(match.group(1))
        subj_val = np.load((self.crossval_subjects_dir / f"subjects_fold_{fold_idx}_val.npy"), allow_pickle=True)
        return subj_val
    
        
    def get_bc_params(self, csv_trainval):
        """ Calculate the bias correction parameters (slope and intercept)
        """
        df = pd.read_csv(csv_trainval)
        if 'DeepBrainNet' in csv_trainval:
            df = df.loc[(df['age']>=45)&(df['age']<90), ]
        else:  # use the validation set for bias correction
            subj_val = self.get_subj_val(csv_trainval)
            df = df.loc[df['dataset_subject'].isin(subj_val)&(df['age']>=45)&(df['age']<90), ]
        df = df.groupby('dataset_subject').apply(lambda x: x.loc[x['age'].idxmin()]).reset_index(drop=True)  # take the cross-sectional samples
        
        x = sm.add_constant(df['age'])
        y = df['age_pred'] - df['age']
        model = sm.OLS(y, x)
        results = model.fit()
        
        slope = results.params['age']
        intercept = results.params['const']
        
        return slope, intercept
        
        
    def apply_bc(self, slope, intercept, csv):
        df = pd.read_csv(csv)
        df_bc = df.copy()
        df_bc['age_pred'] = df['age_pred'] - (slope*df['age'] + intercept)
        return df_bc
    
    
    def perform_bias_correction_for_all(self, suffix='_bc.csv'):
        dict_prediction_csv = self.get_trainval_test_csv_dict()
        for csv_trainval, csv_test in zip(dict_prediction_csv['trainval'], dict_prediction_csv['test']):
            slope, intercept = self.get_bc_params(csv_trainval)
            df_bc = self.apply_bc(slope, intercept, csv_test)
            df_bc.to_csv(csv_test.replace('.csv', suffix), index=False)
            print(f"Bias correction applied to {csv_test}")
            

if __name__ == "__main__":
    crossval_subjects_dir = '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/cross_validation/'
    prediction_roots = [
        'models/2023-12-22_ResNet101/predictions',
        'models/2024-01-16_ResNet101_MLP/predictions',
        'models/2024-02-07_ResNet101_BRAID_warp/predictions',
        'models/2024-02-07_T1wAge_ResNet101/predictions',
        'models/2024-02-12_TSAN_first_stage/predictions',
        'models/2024-02-13_TSAN_second_stage/predictions',
        'models/2024-04-04_DeepBrainNet/predictions',
    ]
    for prediction_root in tqdm(prediction_roots):
        bc = BiasCorrection(prediction_root, crossval_subjects_dir)
        bc.perform_bias_correction_for_all(suffix='_bc.csv')
    