import pandas as pd
from tqdm import tqdm
from pathlib import Path


def roster_brain_age_models():
    dict_models = {
        'WM age nonlinear': {
            'prediction_root': 'models/2024-02-07_ResNet101_BRAID_warp/predictions',
            'col_suffix': '_wm_age_nonlinear',
        },
        'WM age affine': {
            'prediction_root': 'models/2023-12-22_ResNet101/predictions',
            'col_suffix': '_wm_age_affine',
        },
        'GM age (ours)': {
            'prediction_root': 'models/2024-02-07_T1wAge_ResNet101/predictions',
            'col_suffix': '_gm_age_ours',
        },
        'GM age (DeepBrainNet)': {
            'prediction_root': 'models/2024-04-04_DeepBrainNet/predictions',
            'col_suffix': '_gm_age_dbn',
        },
        'GM age (TSAN)': {
            'prediction_root': 'models/2024-02-12_TSAN_first_stage/predictions',
            'col_suffix': '_gm_age_tsan',
        },
    }
    return dict_models


class DataPreparation:
    def __init__(self, dict_models, databank_csv):
        self.dict_models = dict_models
        self.databank = pd.read_csv(databank_csv)
    
    def load_predictions_of_all_models(self, bias_correction=True):
        bc = '_bc' if bias_correction else ''
        
        for i, model in enumerate(self.dict_models.keys()):
            pred_root = Path(self.dict_models[model]['prediction_root'])
            col_suffix = self.dict_models[model]['col_suffix']
            
            if model == 'GM age (DeepBrainNet)':
                pred_csv = pred_root / f'predicted_age_test{bc}.csv'
                df_model = pd.read_csv(pred_csv)
                df_model = df_model.groupby(['dataset','subject','session','sex','age'])['age_pred'].mean().reset_index()
                df_model = df_model.rename(columns={'age_pred': f'age_pred{col_suffix}'})
            else:
                for fold_idx in [1,2,3,4,5]:
                    pred_csv = pred_root / f"predicted_age_fold-{fold_idx}_test{bc}.csv"
                    if fold_idx == 1:
                        df_model = pd.read_csv(pred_csv)
                        df_model = df_model.groupby(['dataset','subject','session','sex','age'])['age_pred'].mean().reset_index()
                        df_model = df_model.rename(columns={'age_pred': f'age_pred{col_suffix}_{fold_idx}'})
                    else:
                        tmp = pd.read_csv(pred_csv)
                        tmp = tmp.groupby(['dataset','subject','session','sex','age'])['age_pred'].mean().reset_index()
                        tmp = tmp.rename(columns={'age_pred': f'age_pred{col_suffix}_{fold_idx}'})
                        df_model = df_model.merge(tmp, on=['dataset','subject','session','sex','age'])
                df_model[f'age_pred{col_suffix}'] = df_model[[f'age_pred{col_suffix}_{fold_idx}' for fold_idx in [1,2,3,4,5]]].mean(axis=1)
            df_model[f'age_pred{col_suffix}_bag'] = df_model[f'age_pred{col_suffix}'] - df_model['age']
            df_model = df_model[['dataset','subject','session','sex','age',f'age_pred{col_suffix}',f'age_pred{col_suffix}_bag']]
            print(f'Loaded data for {model}, shape: {df_model.shape}')
            
            if i == 0:
                df = df_model.copy()
            else:
                df = df.merge(df_model.copy(), on=['dataset','subject','session','sex','age'])
        
        # remove duplicated rows
        df = df.sort_values(by=['dataset', 'subject', 'age', 'session'], ignore_index=True)
        df = df.drop_duplicates(subset=['dataset', 'subject', 'age'], keep='last', ignore_index=True)
        print(f"--------> Predictions loaded. DataFrame shape: {df.shape}")
        return df
    
    def retrieve_diagnosis_label(self, df):
        """ Retrieve diagnosis information from the databank and add as a new column "diagnosis".
        """
        df['diagnosis'] = None
        
        for i, row in tqdm(df.iterrows(), total=len(df.index), desc='Retrieve diagnosis information'):
            loc_filter = (self.databank['dataset']==row['dataset']) & (self.databank['subject']==row['subject']) & ((self.databank['session']==row['session']) | self.databank['session'].isnull())
            if row['dataset'] in ['UKBB']:
                control_label = self.databank.loc[loc_filter, 'control_label'].values[0]
                df.loc[i,'diagnosis'] = 'normal' if control_label == 1 else None
            else:
                df.loc[i,'diagnosis'] = self.databank.loc[loc_filter, 'diagnosis_simple'].values[0]
        
        df['diagnosis'] = df['diagnosis'].replace('dementia', 'AD')
        print(f"--------> Diagnosis labels retrieved. {len(df.loc[df['diagnosis'].isna(),].index)} out of {len(df.index)} do not have diagnosis info.")
        return df
    
    def assign_cn_label(self, df):
        """ Create the following new columns:
        "cn_label": 0.5 for cognitively normal, and has only cognitively normal in his/her diagnosis history.
            1 for all above, plus has at least one following session in which the subject is still cognitively normal.
        "age_last_cn": the age of the last available session of the subject (cn_label >= 0.5).
        "time_to_last_cn": the time (in years) to the "age_last_cn".
        """
        if 'subj' not in df.columns:
            df['subj'] = df['dataset'] + '_' + df['subject']
        
        df['cn_label'] = None
        df['age_last_cn'] = None
        df['time_to_last_cn'] = None

        for subj in df.loc[df['diagnosis']=='normal', 'subj'].unique():
            if len(df.loc[df['subj']==subj, 'diagnosis'].unique())==1:  # there is only 'normal' in diagnosis history
                df.loc[df['subj']==subj, 'cn_label'] = 0.5
                df.loc[df['subj']==subj, 'age_last_cn'] = df.loc[df['subj']==subj, 'age'].max()
                if len(df.loc[df['subj']==subj, 'age'].unique())>=2:  # at least two sessions are available
                    # pick all but the last session (which is uncertain if it progresses to MCI/AD)
                    df.loc[(df['subj']==subj) & (df['age']!=df.loc[df['subj']==subj,'age'].max()), 'cn_label'] = 1
        df['time_to_last_cn'] = df['age_last_cn'] - df['age']

        num_subj_strict = len(df.loc[df['cn_label']==1, 'subj'].unique())
        num_subj_loose = len(df.loc[df['cn_label']>=0.5, 'subj'].unique())
        print(f'--------> Found {num_subj_strict} subjects with strict CN label, and {num_subj_loose} subjects with loose CN label.')
        return df
    
    def mark_progression_subjects_out(self, df):
        """ Create the following columns to the dataframe:
            - "age_AD": the age when the subject was diagnosed with AD for the first time.
            - "time_to_AD": the time (in years) to the first AD diagnosis.
            - "age_MCI": the age when the subject was diagnosed with MCI for the first time.
            - "time_to_MCI": the time (in years) to the first MCI diagnosis.
        Subjects with following characteristics are excluded:
            - the diagnosis of available sessions starts with MCI or AD.
            - the subject turned back to cognitively normal after being diagnosed with MCI or AD.
        """
        df = df.loc[df['diagnosis'].isin(['normal', 'MCI', 'AD']), ].copy()

        if 'subj' not in df.columns:
            df['subj'] = df['dataset'] + '_' + df['subject']
        
        for disease in ['AD', 'MCI']:
            df[f'age_{disease}'] = None
            
            for subj in df.loc[df['diagnosis']==disease, 'subj'].unique():
                include_this_subj = True
                rows_subj = df.loc[df['subj']==subj, ].copy()
                rows_subj = rows_subj.sort_values(by='age')
                if rows_subj.iloc[0]['diagnosis'] != 'normal':
                    include_this_subj = False
                for i in range(len(rows_subj.index)-1):
                    if rows_subj.iloc[i]['diagnosis']==disease and rows_subj.iloc[i+1]['diagnosis']=='normal':
                        include_this_subj = False
                        break
                if include_this_subj:
                    df.loc[df['subj']==subj, f'age_{disease}'] = rows_subj.loc[rows_subj['diagnosis']==disease, 'age'].min()
            df[f'time_to_{disease}'] = df[f'age_{disease}'] - df['age']
            
            num_subj = len(df.loc[df[f'age_{disease}'].notna(), 'subj'].unique())
            print(f'--------> Found {num_subj} subjects with {disease} progression.')
        
        return df


def load_sample_and_symlink_data(dir='reports/figures/2024-05-20_Idea_Figure/data/', ages=[45, 65, 85], n=5, random_state=0):
    """ Load the predictions of all models and form the dataframe. From the dataframe, 
    sample n (or the number of samples available, whichever is larger) samples from each age group. 
    Set up files (in symbolic links) for subsequent visualizations in dir.
    """
    # Load 
    data_prep = DataPreparation(roster_brain_age_models(), '/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv')
    df = data_prep.load_predictions_of_all_models()
    df = data_prep.retrieve_diagnosis_label(df)
    df = data_prep.assign_cn_label(df)
    df = data_prep.mark_progression_subjects_out(df)
    
    for age in ages:
        # sample
        df_sampled = pd.DataFrame()
        filter = (df['age']>=age-2.5) & (df['age']<=age+2.5) & (df['cn_label']>=0.5) & (df['dataset']=='BLSA')
        df_sampled = df.loc[filter, ].sample(n=n, random_state=random_state)
        df_sampled.to_csv(dir+f'sampled_{age}.csv', index=False)
          
        # symlink
        path_databank_t1w = Path('/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_t1w')
        path_databank_dti = Path('/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_dti')
        assert path_databank_t1w.exists() and path_databank_dti.exists(), "Run this script on hickory with GDPR mounted."
        
        for i, row in df_sampled.iterrows():
            folder_target = Path(dir) / str(age) / f"{row['dataset']}_{row['subject']}_{row['session']}"
            folder_target.mkdir(parents=True, exist_ok=True)
            
            # original T1w
            folder_source = path_databank_t1w / row['dataset'] / row['subject'] / row['session'] / 'scan-1'
            files = list(folder_source.glob('*_T1w_brain.nii.gz'))
            assert len(files)==1, f"Found {len(files)} *_T1w.nii.gz files in {folder_source}."
            fn_source = files[0]
            fn_target = folder_target / 'T1w_orig.nii.gz'
            fn_target.symlink_to(fn_source)

            # T1w in MNI152 (affine)
            files = list(folder_source.glob('*_T1w_brain_MNI152_Warped.nii.gz'))
            assert len(files)==1, f"Found {len(files)} *_T1w_brain_MNI152_Warped.nii.gz files in {folder_source}."
            fn_source = files[0]
            fn_target = folder_target / 'T1w_mni152_affine.nii.gz'
            fn_target.symlink_to(fn_source)
            
            # original FA/MD images
            fn_source = path_databank_dti / row['dataset'] / row['subject'] / row['session'] / 'scan-1' / 'dti_fitting' / 'fa_skullstrip.nii.gz'
            if fn_source.is_symlink():
                fn_source = Path(str(fn_source.readlink()).replace('DTI', 'run-'))
            fn_target = folder_target / 'fa_orig.nii.gz'
            fn_target.symlink_to(fn_source)
            
            fn_source = path_databank_dti / row['dataset'] / row['subject'] / row['session'] / 'scan-1' / 'dti_fitting' / 'md_skullstrip.nii.gz'
            if fn_source.is_symlink():
                fn_source = Path(str(fn_source.readlink()).replace('DTI', 'run-'))
            fn_target = folder_target / 'md_orig.nii.gz'
            fn_target.symlink_to(fn_source)
            
            # FA/MD images in MNI152 (affine)
            fn_source = path_databank_dti / row['dataset'] / row['subject'] / row['session'] / 'scan-1' / 'final' / 'fa_skullstrip_MNI152.nii.gz'
            fn_target = folder_target / 'fa_mni152_affine.nii.gz'
            fn_target.symlink_to(fn_source)
            
            fn_source = path_databank_dti / row['dataset'] / row['subject'] / row['session'] / 'scan-1' / 'final' / 'md_skullstrip_MNI152.nii.gz'
            fn_target = folder_target / 'md_mni152_affine.nii.gz'
            fn_target.symlink_to(fn_source)
            
            # FA/MD images in MNI152 (non-rigid)
            fn_source = path_databank_dti / row['dataset'] / row['subject'] / row['session'] / 'scan-1' / 'final' / 'fa_skullstrip_MNI152_warped.nii.gz'
            fn_target = folder_target / 'fa_mni152_syn.nii.gz'
            fn_target.symlink_to(fn_source)
            
            fn_source = path_databank_dti / row['dataset'] / row['subject'] / row['session'] / 'scan-1' / 'final' / 'md_skullstrip_MNI152_warped.nii.gz'
            fn_target = folder_target / 'md_mni152_syn.nii.gz'
            fn_target.symlink_to(fn_source)