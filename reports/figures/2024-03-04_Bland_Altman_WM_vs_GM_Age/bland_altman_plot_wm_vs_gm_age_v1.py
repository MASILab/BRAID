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
        self.marker_size = 5
        self.marker_linewidth = 0
        self.alpha = {
            'scatter': 0.75,
            'scatter_background':0.2,
            'kde': 0.6,
        }
        self.fontfamily = 'DejaVu Sans'
        self.fontsize = {'title': 9, 'label': 9, 'ticks': 9, 'legend': 9}        
    
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
        # df.to_csv('reports/figures/2024-03-04_Bland_Altman_WM_vs_GM_Age/tmp.csv', index=False)  # during development
        
        return df

    def make_bland_altman_plot(self, png, csv='reports/figures/2024-03-04_Bland_Altman_WM_vs_GM_Age/tmp.csv'):
        
        if Path(csv).is_file():
            print('Loading existing csv file for Bland-Altman plot...')
            df = pd.read_csv(csv)
        else:
            print('Preparing dataframe for Bland-Altman plot...')
            df = self.prepare_dataframe_for_bland_altman_plot()
        
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
        sns.scatterplot(
            data=df,
            x='mean',
            y='diff',
            hue='diagnosis',
            s=self.marker_size,
            linewidth=self.marker_linewidth,
            alpha=self.alpha['scatter'],
            ax=ax
        )
        ax.set_xlabel('(WM Age + GM Age) / 2 (years)', fontsize=self.fontsize['label'], fontname=self.fontfamily)
        ax.set_ylabel('WM Age - GM Age (years)', fontsize=self.fontsize['label'], fontname=self.fontfamily)
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
    b.make_bland_altman_plot(png='reports/figures/2024-03-04_Bland_Altman_WM_vs_GM_Age/figs/bland_altman_wm_vs_gm_age_v1.png')