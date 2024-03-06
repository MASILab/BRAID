import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

class BlandAltmanPlotWMGM:
    def __init__(self, wm_pred_root, gm_pred_root, fn_pattern, databank_dti_csv, databank_t1w_csv):
        self.wm_pred_root = wm_pred_root
        self.gm_pred_root = gm_pred_root
        self.fn_pattern = fn_pattern
        self.databank_dti = pd.read_csv(databank_dti_csv)
        self.databank_t1w = pd.read_csv(databank_t1w_csv)

        # Plotting parameters
        self.marker_size = 8
        self.marker_linewidth = 0
        self.alpha = {
            'scatter': 0.75,
            'scatter_background':0.2,
            'kde': 0.6,
        }
        self.fontfamily = 'DejaVu Sans'
        self.fontsize = {'title': 9, 'label': 9, 'ticks': 9, 'legend': 9}
        self.ylim = [-15, 15]
    
    def find_pred_csvs(self, pred_root, fn_pattern):
        csvs = sorted(Path(pred_root).glob(fn_pattern))
        return csvs
    
    def merge_predictions(self, csvs):
        """ Average predictions across scans of the same session.
        Merge predictions from all five folds.
        """
        for i, csv in enumerate(csvs):
            match = re.search(r'fold-(\d+)', csv.name)
            fold_idx = int(match.group(1))
            assert fold_idx == i+1, f"Mismatch between fold index and csv index: {fold_idx} vs {i+1}"
            
            if fold_idx == 1:
                df = pd.read_csv(csv)
                df = df.groupby(['dataset','subject','session','age_gt'])['age_pred'].mean().reset_index()
                df = df.rename(columns={'age_pred': f'age_pred_{fold_idx}'})
            else:
                tmp = pd.read_csv(csv)
                tmp = tmp.groupby(['dataset','subject','session','age_gt'])['age_pred'].mean().reset_index()
                tmp = tmp.rename(columns={'age_pred': f'age_pred_{fold_idx}'})
                df = df.merge(tmp, on=['dataset','subject','session','age_gt'])
        df['age_pred_mean'] = df[['age_pred_1', 'age_pred_2', 'age_pred_3', 'age_pred_4', 'age_pred_5']].mean(axis=1)
        return df[['dataset','subject','session','age_gt','age_pred_mean']].copy()
    
    def collect_diagnosis(self, df, databank):
        df['diagnosis'] = df.apply(lambda row: databank.loc[
            (databank['dataset'] == row['dataset']) &
            (databank['subject'] == row['subject']) &
            ((databank['session'] == row['session']) | (databank['session'].isnull())),
            'diagnosis_simple'].values[0], axis=1)
        return df
    
    def filter_dataframe(self, df, age_min=45, age_max=90, list_diagnosis=['normal', 'MCI', 'dementia']):
        df = df.loc[(df['age_gt']>=age_min)& 
                    (df['age_gt']<=age_max)& 
                    (df['diagnosis'].isin(list_diagnosis)), ]
        return df
    
    def prepare_dataframe_for_bland_altman_plot(self):
        wm_pred_csvs = self.find_pred_csvs(self.wm_pred_root, self.fn_pattern)
        gm_pred_csvs = self.find_pred_csvs(self.gm_pred_root, self.fn_pattern)
        
        df_wm = self.merge_predictions(wm_pred_csvs)
        df_gm = self.merge_predictions(gm_pred_csvs)
        
        df_wm = self.collect_diagnosis(df_wm, self.databank_dti)
        df_gm = self.collect_diagnosis(df_gm, self.databank_t1w)

        df_wm = self.filter_dataframe(df_wm)
        df_gm = self.filter_dataframe(df_gm)
        
        df = df_wm.merge(df_gm[['dataset','subject','session','age_pred_mean']], on=['dataset','subject','session'], suffixes=('_wm', '_gm'))
        df['diff'] = df['age_pred_mean_wm'] - df['age_pred_mean_gm']
        df['mean'] = (df['age_pred_mean_wm'] + df['age_pred_mean_gm']) / 2
        
        df['Category'] = None
        df['dsubj'] = df['dataset'] + '_' + df['subject']
        # first session of the disease-free subjects (with at least one follow-up session)
        for subj in df.loc[df['diagnosis']=='normal', 'dsubj'].unique():
            if (len(df.loc[df['dsubj']==subj, 'diagnosis'].unique()) == 1) and (len(df.loc[df['dsubj']==subj, 'age_gt'].unique()) >= 2):
                df.loc[(df['dsubj'] == subj) & 
                       (df['age_gt'] == df.loc[df['dsubj'] == subj, 'age_gt'].min()), 'Category'] = 'Earliest sample of CN subjects (who stayed CN in all follow-ups)'
                continue
        # session after which the subject converted to MCI or dementia
        for subj in df.loc[df['diagnosis'].isin(['MCI', 'dementia']), 'dsubj'].unique():
            rows_subj = df.loc[df['dsubj']==subj, ].copy()
            if 'normal' in rows_subj['diagnosis'].values:
                rows_subj = rows_subj.sort_values(by='age_gt')
                for i in range(len(rows_subj.index)-1):
                    if (rows_subj.iloc[i]['diagnosis'] == 'normal') & (rows_subj.iloc[i+1]['diagnosis'] in ['MCI', 'dementia']):
                        df.loc[(df['dsubj'] == subj) & 
                               (df['age_gt'] == rows_subj.iloc[i]['age_gt']), 'Category'] = 'CN subjects (converted to MCI/dementia in the next follow-up)'
                        break
        # the last session of the MCI subjects
        for subj in df.loc[df['diagnosis']=='MCI', 'dsubj'].unique():
            df.loc[(df['dsubj'] == subj) &
                   (df['age_gt'] == df.loc[(df['dsubj'] == subj)&(df['diagnosis']=='MCI'), 'age_gt'].max()), 'Category'] = 'Last sample of MCI subjects'
        # the last session of the dementia subjects
        for subj in df.loc[df['diagnosis']=='dementia', 'dsubj'].unique():
            df.loc[(df['dsubj'] == subj) &
                   (df['age_gt'] == df.loc[(df['dsubj'] == subj)&(df['diagnosis']=='dementia'), 'age_gt'].max()), 'Category'] = 'Last sample of dementia subjects'
        df.to_csv('reports/figures/2024-03-04_Bland_Altman_WM_vs_GM_Age/tmp.csv', index=False)  # during development
        
        return df

    def make_bland_altman_plot(self, png, csv='reports/figures/2024-03-04_Bland_Altman_WM_vs_GM_Age/tmp.csv'):
        
        if Path(csv).is_file():
            print('Loading existing csv file for Bland-Altman plot...')
            df = pd.read_csv(csv)
        else:
            print('Preparing dataframe for Bland-Altman plot...')
            df = self.prepare_dataframe_for_bland_altman_plot()
        df = df.dropna(subset=['Category'])

        fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.5))
        sns.scatterplot(
            data=df,
            x='mean',
            y='diff',
            hue='Category',
            style='Category',
            s=self.marker_size,
            linewidth=self.marker_linewidth,
            alpha=self.alpha['scatter'],
            ax=ax
        )
        ax.axhline(y=0, linestyle='-', linewidth=1, color='k', alpha=0.25)

        ax.set_xlabel('(WM Age + GM Age) / 2 (years)', fontsize=self.fontsize['label'], fontname=self.fontfamily)
        ax.set_ylabel('WM Age - GM Age (years)', fontsize=self.fontsize['label'], fontname=self.fontfamily)
        ax.set_ylim(self.ylim[0], self.ylim[1])
        ax.legend(prop={'size': self.fontsize['legend'], 'family': self.fontfamily})
        # Save figure
        Path(png).parent.mkdir(parents=True, exist_ok=True)
        fig.subplots_adjust(hspace=0,wspace=0.05)
        fig.savefig(png,
                    dpi=300,
                    bbox_inches='tight')


if __name__ == '__main__':
    b = BlandAltmanPlotWMGM(
        wm_pred_root='models/2024-02-07_ResNet101_BRAID_warp/predictions', 
        gm_pred_root='models/2024-02-07_T1wAge_ResNet101/predictions', 
        fn_pattern='predicted_age_fold-*_test_bc.csv', 
        databank_dti_csv='/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti_v2.csv', 
        databank_t1w_csv='/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_t1w_v2.csv',
    )
    b.make_bland_altman_plot(png='reports/figures/2024-03-04_Bland_Altman_WM_vs_GM_Age/figs/bland_altman_wm_vs_gm_age_v2.png')