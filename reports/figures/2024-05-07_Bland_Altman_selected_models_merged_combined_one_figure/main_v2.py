import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from pathlib import Path

def roster_brain_age_models():
    dict_models = {
        'WM age nonlinear': {
            'prediction_root': 'models/2024-02-07_ResNet101_BRAID_warp/predictions',
            'col_suffix': '_wm_age_nonlinear',
            'yaxis_label': 'WM age (nonrigid)\nbrain age gap (years)'
        },
        'WM age affine': {
            'prediction_root': 'models/2023-12-22_ResNet101/predictions',
            'col_suffix': '_wm_age_affine',
            'yaxis_label': 'WM age (affine)\nbrain age gap (years)'
        },
        'GM age (ours)': {
            'prediction_root': 'models/2024-02-07_T1wAge_ResNet101/predictions',
            'col_suffix': '_gm_age_ours',
            'yaxis_label': 'GM age (ours)\nbrain age gap (years)'
        },
        'GM age (DeepBrainNet)': {
            'prediction_root': 'models/2024-04-04_DeepBrainNet/predictions',
            'col_suffix': '_gm_age_dbn',
            'yaxis_label': 'GM age (DBN)\nbrain age gap (years)'
        },
        'GM age (TSAN)': {
            'prediction_root': 'models/2024-02-12_TSAN_first_stage/predictions',
            'col_suffix': '_gm_age_tsan',
            'yaxis_label': 'GM age (TSAN)\nbrain age gap (years)'
        },
    }
    return dict_models


class BlandAltman:
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

    def plot_bland_altman(self, df, cnstar_threshold, png):
        # hyperparameters
        fontsize = 8.5
        fontfamily = 'DejaVu Sans'
        xlim = [25, 110]
        ylim = [-28, 28]
        xticks = [45, 60, 75, 90]
        yticks = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
        marker_props = {'size': 2, 'linewidth': 0}
        colors = {
            'cmap_main': 'RdPu',
            'marker_main': 'tab:red',
            'marker_outside': 'tab:gray',
            'hvline': 'k',
        }
        alphas = {
            'scatter_inside': 0.5,
            'scatter_outside': 0.2,
            'kde': 0.5,
            'hvline': 0.25,
        }
        fig = plt.figure(figsize=(6.5, 7.5), tight_layout=True)
        gs = gridspec.GridSpec(nrows=6, ncols=5, wspace=0, hspace=0, width_ratios=[0.3, 1, 1, 1, 1], height_ratios=[1, 1, 1, 1, 1, 0.2])
        
        row_idx = 0
        for model_name, model_dict in self.dict_models.items():
            # Left column: y axis label
            ax = fig.add_subplot(gs[row_idx, 0])
            ax.text(0, 0.5, model_dict['yaxis_label'], fontsize=fontsize, fontfamily=fontfamily, ha='center', va='center', rotation='vertical', transform=ax.transAxes)
            ax.axis('off')
            
            # 1/4: Cognitively Normal
            ax = fig.add_subplot(gs[row_idx, 1])
            data = df.loc[df['cn_label']>=0.5, ].copy()
            age_mask = data['age'].between(45, 90)
            sns.scatterplot(
                data=data.loc[~age_mask, ], x='age', y=f'age_pred{model_dict["col_suffix"]}_bag',
                s=marker_props['size'], linewidth=marker_props['linewidth'], color=colors['marker_outside'], alpha=alphas['scatter_outside'],
                ax=ax
            )
            sns.scatterplot(
                data=data.loc[age_mask, ], x='age', y=f'age_pred{model_dict["col_suffix"]}_bag',
                s=marker_props['size'], linewidth=marker_props['linewidth'], color=colors['marker_main'], alpha=alphas['scatter_inside'],
                ax=ax
            )
            sns.kdeplot(
                data=data.loc[age_mask, ], x='age', y=f'age_pred{model_dict["col_suffix"]}_bag',
                fill=True, levels=10, cut=1, cmap=colors['cmap_main'], alpha=alphas['kde'],
                ax=ax,
            )
            
            ax.axhline(y=0, linestyle='-', linewidth=1, color=colors['hvline'], alpha=alphas['hvline'])
            ax.axvline(x=45, linestyle='--', linewidth=1, color=colors['hvline'], alpha=alphas['hvline'])
            ax.axvline(x=90, linestyle='--', linewidth=1, color=colors['hvline'], alpha=alphas['hvline'])
            ax.text(0.04, 0.96, 'CN', fontsize=fontsize, fontfamily=fontfamily, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
            mae = data.loc[age_mask, f"age_pred{model_dict['col_suffix']}_bag"].abs().mean()
            ax.text(0.5, 0, f'MAE={mae:.2f}', fontsize=fontsize, fontfamily=fontfamily, transform=ax.transAxes,
                    ha='center', va='bottom')
            
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim(left=xlim[0], right=xlim[1])
            if row_idx == 4:
                ax.set_xticks(xticks)
                ax.tick_params(axis='x', which='both', direction='out', length=4, pad=2, labelsize=fontsize, labelfontfamily=fontfamily)
            else:
                ax.set_xticks([])

            ax.set_ylim(bottom=ylim[0], top=ylim[1])
            ax.set_yticks(yticks)
            ax.tick_params(axis='y', which='both', direction='inout', length=2, pad=1, labelsize=fontsize, labelfontfamily=fontfamily)
            
            
            # CN*
            ax = fig.add_subplot(gs[row_idx, 2])
            data = df.loc[df['time_to_MCI']>cnstar_threshold, ].copy()
            age_mask = data['age'].between(45, 90)
            sns.scatterplot(
                data=data.loc[~age_mask, ], x='age', y=f'age_pred{model_dict["col_suffix"]}_bag',
                s=marker_props['size'], linewidth=marker_props['linewidth'], color=colors['marker_outside'], alpha=alphas['scatter_outside'],
                ax=ax
            )
            sns.scatterplot(
                data=data.loc[age_mask, ], x='age', y=f'age_pred{model_dict["col_suffix"]}_bag',
                s=marker_props['size'], linewidth=marker_props['linewidth'], color=colors['marker_main'], alpha=alphas['scatter_inside'],
                ax=ax
            )
            sns.kdeplot(
                data=data.loc[age_mask, ], x='age', y=f'age_pred{model_dict["col_suffix"]}_bag',
                fill=True, levels=10, cut=1, cmap=colors['cmap_main'], alpha=alphas['kde'],
                ax=ax,
            )
            
            ax.axhline(y=0, linestyle='-', linewidth=1, color=colors['hvline'], alpha=alphas['hvline'])
            ax.axvline(x=45, linestyle='--', linewidth=1, color=colors['hvline'], alpha=alphas['hvline'])
            ax.axvline(x=90, linestyle='--', linewidth=1, color=colors['hvline'], alpha=alphas['hvline'])
            ax.text(0.04, 0.96, 'CN*', fontsize=fontsize, fontfamily=fontfamily, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
            mae = data.loc[age_mask, f"age_pred{model_dict['col_suffix']}_bag"].abs().mean()
            ax.text(0.5, 0, f'MAE={mae:.2f}', fontsize=fontsize, fontfamily=fontfamily, transform=ax.transAxes,
                    ha='center', va='bottom')
            
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim(left=xlim[0], right=xlim[1])
            if row_idx == 4:
                ax.set_xticks(xticks)
                ax.tick_params(axis='x', which='both', direction='out', length=4, pad=2, labelsize=fontsize, labelfontfamily=fontfamily)
            else:
                ax.set_xticks([])
            ax.set_ylim(bottom=ylim[0], top=ylim[1])
            ax.set_yticks(yticks)
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', direction='inout', length=2)
            
            
            # MCI
            ax = fig.add_subplot(gs[row_idx, 3])
            data = df.loc[df['diagnosis']=='MCI', ].copy()
            age_mask = data['age'].between(45, 90)
            sns.scatterplot(
                data=data.loc[~age_mask, ], x='age', y=f'age_pred{model_dict["col_suffix"]}_bag',
                s=marker_props['size'], linewidth=marker_props['linewidth'], color=colors['marker_outside'], alpha=alphas['scatter_outside'],
                ax=ax
            )
            sns.scatterplot(
                data=data.loc[age_mask, ], x='age', y=f'age_pred{model_dict["col_suffix"]}_bag',
                s=marker_props['size'], linewidth=marker_props['linewidth'], color=colors['marker_main'], alpha=alphas['scatter_inside'],
                ax=ax
            )
            sns.kdeplot(
                data=data.loc[age_mask, ], x='age', y=f'age_pred{model_dict["col_suffix"]}_bag',
                fill=True, levels=10, cut=1, cmap=colors['cmap_main'], alpha=alphas['kde'],
                ax=ax,
            )
            
            ax.axhline(y=0, linestyle='-', linewidth=1, color=colors['hvline'], alpha=alphas['hvline'])
            ax.axvline(x=45, linestyle='--', linewidth=1, color=colors['hvline'], alpha=alphas['hvline'])
            ax.axvline(x=90, linestyle='--', linewidth=1, color=colors['hvline'], alpha=alphas['hvline'])
            ax.text(0.04, 0.96, 'MCI', fontsize=fontsize, fontfamily=fontfamily, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
            mae = data.loc[age_mask, f"age_pred{model_dict['col_suffix']}_bag"].abs().mean()
            ax.text(0.5, 0, f'MAE={mae:.2f}', fontsize=fontsize, fontfamily=fontfamily, transform=ax.transAxes,
                    ha='center', va='bottom')
            
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim(left=xlim[0], right=xlim[1])
            if row_idx == 4:
                ax.set_xticks(xticks)
                ax.tick_params(axis='x', which='both', direction='out', length=4, pad=2, labelsize=fontsize, labelfontfamily=fontfamily)
            else:
                ax.set_xticks([])
            ax.set_ylim(bottom=ylim[0], top=ylim[1])
            ax.set_yticks(yticks)
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', direction='inout', length=2)
            
            
            # AD
            ax = fig.add_subplot(gs[row_idx, 4])
            data = df.loc[df['diagnosis']=='AD', ].copy()
            age_mask = data['age'].between(45, 90)
            sns.scatterplot(
                data=data.loc[~age_mask, ], x='age', y=f'age_pred{model_dict["col_suffix"]}_bag',
                s=marker_props['size'], linewidth=marker_props['linewidth'], color=colors['marker_outside'], alpha=alphas['scatter_outside'],
                ax=ax
            )
            sns.scatterplot(
                data=data.loc[age_mask, ], x='age', y=f'age_pred{model_dict["col_suffix"]}_bag',
                s=marker_props['size'], linewidth=marker_props['linewidth'], color=colors['marker_main'], alpha=alphas['scatter_inside'],
                ax=ax
            )
            sns.kdeplot(
                data=data.loc[age_mask, ], x='age', y=f'age_pred{model_dict["col_suffix"]}_bag',
                fill=True, levels=10, cut=1, cmap=colors['cmap_main'], alpha=alphas['kde'],
                ax=ax,
            )
            
            ax.axhline(y=0, linestyle='-', linewidth=1, color=colors['hvline'], alpha=alphas['hvline'])
            ax.axvline(x=45, linestyle='--', linewidth=1, color=colors['hvline'], alpha=alphas['hvline'])
            ax.axvline(x=90, linestyle='--', linewidth=1, color=colors['hvline'], alpha=alphas['hvline'])
            ax.text(0.04, 0.96, 'AD', fontsize=fontsize, fontfamily=fontfamily, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
            mae = data.loc[age_mask, f"age_pred{model_dict['col_suffix']}_bag"].abs().mean()
            ax.text(0.5, 0, f'MAE={mae:.2f}', fontsize=fontsize, fontfamily=fontfamily, transform=ax.transAxes,
                    ha='center', va='bottom')
            
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim(left=xlim[0], right=xlim[1])
            if row_idx == 4:
                ax.set_xticks(xticks)
                ax.tick_params(axis='x', which='both', direction='out', length=4, pad=2, labelsize=fontsize, labelfontfamily=fontfamily)
            else:
                ax.set_xticks([])
            ax.set_ylim(bottom=ylim[0], top=ylim[1])
            ax.set_yticks(yticks)
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', direction='inout', length=2)
            
            row_idx += 1
        
        # xlabel
        ax = fig.add_subplot(gs[row_idx, 1:])
        ax.text(0.5, 0, 'chronological age (years)', fontsize=fontsize, fontfamily=fontfamily, ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        
        fig.savefig(png, dpi=600)

if __name__ == '__main__':
    bland_altman = BlandAltman(roster_brain_age_models(), '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv')
    df = bland_altman.load_predictions_of_all_models()
    df = bland_altman.retrieve_diagnosis_label(df)
    df = bland_altman.assign_cn_label(df)
    df = bland_altman.mark_progression_subjects_out(df)
    
    cnstar_threshold_choices = [0, 1, 2, 3, 4, 5]
    for t in cnstar_threshold_choices:
        bland_altman.plot_bland_altman(df, t, f'reports/figures/2024-05-07_Bland_Altman_selected_models_merged_combined_one_figure/figs/v2/brain_age_models_bland_altman_cnstar-{t}.png')