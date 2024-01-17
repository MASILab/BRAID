import pdb
import torch
import subprocess
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path, PosixPath
from braid.models import get_the_resnet_model


def load_trained_model(model_name, mlp_hidden_layer_sizes, feature_vector_length, path_pth, device='cuda'):
    # model architecture
    model = get_the_resnet_model(
        model_name = model_name,
        feature_vector_length = feature_vector_length,
        MLP_hidden_layer_sizes = mlp_hidden_layer_sizes,
    )
    
    # load model weights
    checkpoint = torch.load(path_pth)
    model.load_state_dict(checkpoint)
    
    if device == 'cuda':
        model = model.to(torch.device('cuda'))
    
    print(f"Trained model loaded in {device}")
    return model


class AgePredictionEvaluator():
    def __init__(self, prediction_csv):
        self.prediction_csv = Path(prediction_csv)
        self.df = pd.read_csv(prediction_csv)
        
        # inspect if the csv has the required columns for analysis
        columns_to_check = [
            'dataset','subject','session','scan',
            'sex','race_simple','age','control_label',
            'dataset_subject','age_gt','age_pred',
        ]
        for col in columns_to_check:
            if col not in self.df.columns: print(f'Column {col} not found in the csv. Some analysis may not work.')
        
        # sanity check: age_gt and age should have same values across rows
        if ((self.df['age'] - self.df['age_gt']).abs() >= 0.001).any():
            raise ValueError("Possible index issues in inference: age_gt and age do not match!")

    def get_a_subset_of_dataframe(
        self, 
        dataset: list[str] | str = 'all', 
        sex: list[str] | str = 'all',
        race: list[str] | str = 'all', 
        age_min: int = 45,
        age_max: int = 90,
        age_inverse_selection: bool = False,
        control_label: list[int] | str = 'all',
        save_csv: str | PosixPath | None = None,
    ):
        """filter the dataframe and return a subset of it"""

        # filter by dataset
        if type(dataset) == str:
            if dataset == 'all':
                mask = pd.Series([True] * len(self.df.index))
            else:
                raise ValueError("dataset must be either 'all' or a list")
        else:
            mask = self.df['dataset'].isin(dataset)
        
        # fiter by sex
        if type(sex) == str:
            if sex == 'all':
                pass
            else:
                raise ValueError("sex must be either 'all' or a list")
        else:
            mask = mask & self.df['sex'].isin(sex)
            
        # filter by race
        if type(race) == str:
            if race == 'all':
                pass
            else:
                raise ValueError("race must be either 'all' or a list")
        else:
            mask = mask & self.df['race_simple'].isin(race)
        
        # filter by age
        if age_inverse_selection:
            mask = mask & ((self.df['age'] < age_min) | (self.df['age'] >= age_max))
        else:
            mask = mask & ((self.df['age'] >= age_min) & (self.df['age'] < age_max))
        
        # filter by control_label
        if type(control_label) == str:
            if control_label == 'all':
                pass
            else:
                raise ValueError("control_label must be either 'all' or a list")
        else:
            mask = mask & self.df['control_label'].isin(control_label)
        
        df_subset = self.df.loc[mask, ]
        if save_csv is not None:
            subprocess.run(['mkdir', '-p', str(Path(save_csv).parent)])
            df_subset.to_csv(save_csv)
        
        return df_subset
        
    def calculate_mae(self, df_subset):
        """calculate mean absolute error (MAE)"""
        return (df_subset['age_gt'] - df_subset['age_pred']).abs().mean()
    
    def report_mae_across_subsets(
        self,
        save_report_in_csv: str | PosixPath,
        labels_unfold: list[str] = ['dataset', 'age_opt'],
    ):
        """report MAE across subsets of the dataframe
        
        Args:
        save_report_in_csv: path to save the report in csv format.
        labels_unfold: list of labels to unfold. For example, if 'dataset' is in the list,
            then the report will include MAE for each dataset, in addition to the overall MAE.
        """
        list_selections_dataset = ['all'] 
        if 'dataset' in labels_unfold:
            list_selections_dataset += [[dataset] for dataset in self.df['dataset'].unique()]
        
        list_selections_sex = ['all']
        if 'sex' in labels_unfold:
            list_selections_sex += [[sex] for sex in self.df['sex'].unique()]

        list_selections_race = ['all']
        if 'race' in labels_unfold:
            list_selections_race += [[race] for race in self.df['race_simple'].unique()]
        
        if 'age_opt' in labels_unfold:
            list_selections_age_opt = [[45, 90, False], [45, 90, True]]
        else:
            list_selections_age_opt = [[45, 90, False]]
        
        list_selections_control_label = [[control_label] for control_label in self.df['control_label'].unique()]
        
        # calcualte MAE for each subset and save to dataframe
        list_df_dataset = []
        list_df_sex = []
        list_df_race = []
        list_df_age_min = []
        list_df_age_max = []
        list_df_age_inverse_selection = []
        list_df_control_label = []
        list_df_mae = []

        for dataset in list_selections_dataset:
            print(f"Calculating MAE for dataset = {dataset}")
            for sex in list_selections_sex:
                for race in list_selections_race:
                    for age_opt in list_selections_age_opt:
                        age_min = age_opt[0]
                        age_max = age_opt[1]
                        age_inverse_selection = age_opt[2]
                        for control_label in list_selections_control_label:
                            df_subset = self.get_a_subset_of_dataframe(
                                dataset = dataset,
                                sex = sex,
                                race = race,
                                age_min = age_min,
                                age_max = age_max,
                                age_inverse_selection = age_inverse_selection,
                                control_label = control_label
                            )
                            mae = self.calculate_mae(df_subset = df_subset)
                            
                            list_df_dataset.append(dataset)
                            list_df_sex.append(sex)
                            list_df_race.append(race)
                            list_df_age_min.append(age_min)
                            list_df_age_max.append(age_max)
                            list_df_age_inverse_selection.append(age_inverse_selection)
                            list_df_control_label.append(control_label)
                            list_df_mae.append(mae)
        d = {
            'dataset': list_df_dataset,
            'sex': list_df_sex,
            'race': list_df_race,
            'age_min': list_df_age_min,
            'age_max': list_df_age_max,
            'age_inverse_selection': list_df_age_inverse_selection,
            'control_label': list_df_control_label,
            'mae': list_df_mae,
        }
        df_report = pd.DataFrame(data = d)
        subprocess.run(['mkdir', '-p', str(Path(save_report_in_csv).parent)])
        df_report.to_csv(save_report_in_csv, index = False)

    def generate_bland_altman_plot(
        self,
        databank_csv: str | PosixPath,
        save_png: str | PosixPath,
        display_all_ages: bool = False,
    ):
        df_databank = pd.read_csv(databank_csv)
        df = self.df.copy()
        
        # retrieve the diagnosis information
        df['diagnosis'] = df.apply(lambda row: df_databank.loc[
            (df_databank['dataset'] == row['dataset']) &
            (df_databank['subject'] == row['subject']) &
            ((df_databank['session'] == row['session']) | (df_databank['session'].isnull())),
            'diagnosis_simple'].values[0], axis=1)

        # prepare data for bland-altman plot
        df['age_diff'] = df['age_pred'] - df['age_gt']
        df['testing set'] = np.where(df['dataset'] == 'ICBM', 'external dataset', 'internal dataset')
        data_healthy = df.loc[df['control_label']==1, ]
        data_patient = df.loc[(df['diagnosis'] != 'normal') & (df['diagnosis'].notnull()), ]

        # plot settings
        figsize = (6.5, 3.5)
        marker_size = 5
        marker_linewidth = 0
        alpha = {
            'scatter_marker_outside': 0.2,
            'scatter_marker_inside': 1,
            'kde': 0.8,
        }

        title_names = ['healthy', 'patient']
        fontfamily = 'Ubuntu Condensed'
        fontsize = {'title': 9, 'label': 11, 'ticks': 11, 'legend': 9}
        
        sns.set_palette("tab10")
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharex=True, sharey=True)

        # healthy
        age_mask = (data_healthy['age_gt'] >= 45) & (data_healthy['age_gt'] < 90)
        age_mask_inv = ~age_mask
        if display_all_ages:
            sns.scatterplot(
                data=data_healthy.loc[age_mask_inv, ],
                x='age_gt',
                y='age_diff',
                s=marker_size,
                linewidth=marker_linewidth,
                color="tab:gray",
                alpha=alpha['scatter_marker_outside'],
                ax=axes[0]
            )
        sns.scatterplot(
            data=data_healthy.loc[age_mask, ],
            x='age_gt',
            y='age_diff',
            s=marker_size,
            hue='testing set',
            linewidth=marker_linewidth,
            ax=axes[0]
        )
        sns.kdeplot(
            data=data_healthy.loc[age_mask, ],
            x='age_gt',
            y='age_diff',
            hue='testing set',
            fill=True,
            levels=10,
            cut=1,
            alpha=alpha['kde'],
            ax=axes[0]
        )

        # patient
        age_mask = (data_patient['age_gt'] >= 45) & (data_patient['age_gt'] < 90)
        age_mask_inv = ~age_mask
        if display_all_ages:
            sns.scatterplot(
                data=data_patient.loc[age_mask_inv, ],
                x='age_gt',
                y='age_diff',
                s=marker_size,
                color="tab:gray",
                alpha=alpha['scatter_marker_outside'],
                linewidth=marker_linewidth,
                ax=axes[1]
            )
        sns.scatterplot(
            data=data_patient.loc[age_mask, ],
            x='age_gt',
            y='age_diff',
            s=marker_size,
            hue='testing set',
            linewidth=marker_linewidth,
            ax=axes[1]
        )
        sns.kdeplot(
            data=data_patient.loc[age_mask, ],
            x='age_gt',
            y='age_diff',
            hue='testing set',
            fill=True,
            levels=10,
            cut=1,
            alpha=alpha['kde'],
            ax=axes[1]
        )

        # global adjustments
        for i in range(2):
            axes[i].axhline(y=0, linestyle='-',linewidth=1, color='k', alpha=0.25)
            axes[i].axvline(x=45, linestyle='--',linewidth=1, color='k', alpha=0.25)
            axes[i].axvline(x=90, linestyle='--',linewidth=1, color='k', alpha=0.25, label='age range used for training')
            axes[i].legend(prop={'size': fontsize['legend'], 'family': fontfamily})

            axes[i].set_xlabel('chronological age (years)', fontsize=fontsize['label'], fontname=fontfamily)
            axes[i].set_ylabel('predicted age - chronological age (years)', fontsize=fontsize['label'], fontname=fontfamily)
        
            axes[i].text(0.05, 0.95, 
                        title_names[i], 
                        fontsize=fontsize['title'],
                        fontfamily=fontfamily,
                        transform=axes[i].transAxes,
                        verticalalignment='top', 
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
            
            axes[i].tick_params(axis='both', direction='out', length=2)
            axes[i].set_xlim(25, 110)
            axes[i].set_ylim(-25, 25)
            axes[i].set_xticks([30, 45, 60, 75, 90, 105])
            axes[i].set_yticks([-15, -10, -5, 0, 5, 10, 15])

        fig.subplots_adjust(hspace=0,wspace=0.1)
        fig.savefig(save_png,
                    dpi=600,
                    bbox_inches='tight')

    def prepare_dataframe_for_scan_rescan_reproducibility_test(
        self,
        databank_csv: str | PosixPath,
        age_min: int = 45,
        age_max: int = 90,
        cross_sectional: bool = True,
        save_csv: str | PosixPath | None = None,
    ):
        """
        Prepare a dataframe for scan-rescan reproducibility test.
        Should contain columns:
            dataset, subject, session, diagnosis, predicted_age_first_scan, predicted_age_second_scan, predicted_age_diff 
        """
        df = self.df.copy()
        df = df.loc[(df['age_gt'] >= age_min) & (df['age_gt'] < age_max), ]
        
        # retrieve the diagnosis information
        df_databank = pd.read_csv(databank_csv)
        df['diagnosis'] = df.apply(lambda row: df_databank.loc[
            (df_databank['dataset'] == row['dataset']) &
            (df_databank['subject'] == row['subject']) &
            ((df_databank['session'] == row['session']) | (df_databank['session'].isnull())),
            'diagnosis_simple'].values[0], axis=1)
        
        # collecting values for the new dataframe
        list_df_dataset = []
        list_df_subject = []
        list_df_session = []
        list_df_diagnosis = []
        list_df_predicted_age_first_scan = []
        list_df_predicted_age_second_scan = []
        list_df_predicted_age_diff = []
        list_df_predicted_age_diff_abs = []

        df['ds_session'] = df['dataset'] + '_' + df['subject'] + '_' + df['session']
        for ds_session in df['ds_session'].unique():
            predicted_ages = df.loc[df['ds_session']==ds_session, 'age_pred'].values
            if len(predicted_ages) < 2:
                continue
            
            list_df_dataset.append(df.loc[df['ds_session']==ds_session, 'dataset'].values[0])
            list_df_subject.append(df.loc[df['ds_session']==ds_session, 'subject'].values[0])
            list_df_session.append(df.loc[df['ds_session']==ds_session, 'session'].values[0])
            list_df_diagnosis.append(df.loc[df['ds_session']==ds_session, 'diagnosis'].values[0])
            list_df_predicted_age_first_scan.append(predicted_ages[0])
            list_df_predicted_age_second_scan.append(predicted_ages[1])
            list_df_predicted_age_diff.append(predicted_ages[1] - predicted_ages[0])
            list_df_predicted_age_diff_abs.append(abs(predicted_ages[1] - predicted_ages[0]))
        
        d = {
            'dataset': list_df_dataset,
            'subject': list_df_subject,
            'session': list_df_session,
            'diagnosis': list_df_diagnosis,
            'predicted_age_first_scan': list_df_predicted_age_first_scan,
            'predicted_age_second_scan': list_df_predicted_age_second_scan,
            'predicted_age_diff': list_df_predicted_age_diff,
            'predicted_age_diff_abs': list_df_predicted_age_diff_abs,
        }
        df = pd.DataFrame(data = d)
        
        if cross_sectional:
            df = df.loc[df.groupby('subject')['predicted_age_diff_abs'].idxmin()]  # consider how it will introduce bias
            
        if save_csv is not None:
            subprocess.run(['mkdir', '-p', str(Path(save_csv).parent)])
            df.to_csv(save_csv, index=False)

        return df
    
    def generate_scan_rescan_raincloud_plot(
        self,
        save_png: str | PosixPath,
        databank_csv: str | PosixPath,
        age_min: int = 45,
        age_max: int = 90,
        cross_sectional: bool = True,
    ):
        df = self.prepare_dataframe_for_scan_rescan_reproducibility_test(databank_csv, age_min, age_max, cross_sectional)
        df['group'] = df['diagnosis'].apply(lambda x: 'normal' if x == 'normal' else 'patient' if pd.notnull(x) else None)        

        # plot settings
        figsize = (4, 3)
        dpi = 300
        fontfamily = 'Ubuntu Condensed'
        fontsize = {'title': 9, 'label': 11, 'ticks': 11, 'legend': 9}
        bw_adjust = 0.5
        marker_size = 2
        jitter = 0.2 # strip plot
        strip_shift = 0.25
        alpha_scatter = 0.5
        sns.set_style('white')
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)

        # Violin plot
        ax = sns.violinplot(
            data = df,
            x = 'group',
            y = 'predicted_age_diff_abs',
            order = ['normal', 'patient'],
            cut=0,
            density_norm='area',
            width=0.5,
            inner=None,
            bw_adjust=bw_adjust,
            color='#396ef7',
            saturation=1,
            linewidth=0,
            ax=ax,
        )
        
        # Clip the right half of each violin.
        for item in ax.collections:
            x0, y0, width, height = item.get_paths()[0].get_extents().bounds
            item.set_clip_path(plt.Rectangle((x0, y0), width/2, height, transform=ax.transData))
    
        # Create strip plots
        num_items = len(ax.collections)
        ax = sns.stripplot(
            data = df,
            x = 'group',
            y = 'predicted_age_diff_abs',
            order = ['normal', 'patient'],
            jitter = jitter,
            alpha = alpha_scatter,
            color ='#396ef7',
            size = marker_size,
            ax = ax,
        )
        # Shift each strip plot strictly below the correponding volin.
        for item in ax.collections[num_items:]:
            item.set_offsets(item.get_offsets() + (strip_shift, 0))

        # Create narrow boxplots on top of the corresponding violin and strip plots, with thick lines, the mean values, without the outliers.
        ax = sns.boxplot(
            data = df,
            x = 'group',
            y = 'predicted_age_diff_abs',
            order = ['normal', 'patient'],
            width = 0.08,
            showfliers = False,
            boxprops = dict(facecolor=(0,0,0,0),
                            linewidth=1, zorder=2),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1),
            medianprops=dict(color= '#ff8121', 
                             linewidth=1.5),
            ax = ax,
        )

        ax.grid(linestyle=':', linewidth=0.5)
        ax.set_xlabel('')        
        ax.set_ylabel('Scan-rescan absolute difference (years)', fontsize=fontsize['label'], fontname=fontfamily)
        ax.tick_params(labelsize=fontsize['ticks'], labelfontfamily=fontfamily)
        ax.set_ylim(bottom=0, top=8)
        ax.set_yticks([0,1,2,3,4,5,6,7,8])
        fig.savefig(save_png, dpi = dpi)

# scan-rescan reproducibility
# cognitive score correlation

evaluator = AgePredictionEvaluator(prediction_csv='models/2023-12-22_ResNet101/predictions/predicted_age_fold-1.csv')
evaluator.generate_scan_rescan_raincloud_plot(dataframe=None, save_png='test_scan_rescan.png')
    