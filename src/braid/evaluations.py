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
        df['testing set'] = np.where(df['dataset'] == 'ICBM', 'external', 'internal')
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
        
        sns.set_palette("deep")  #TODO: consider change the color palette

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
        # TODO: kde params
        sns.kdeplot(
            data=data_patient.loc[age_mask, ],
            x='age_gt',
            y='age_diff',
            hue='testing set',
            fill=True,
            levels=10,
            cut=0,
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

# scan-rescan reproducibility
# bland-altman plot
# cognitive score correlation

evaluator = AgePredictionEvaluator(prediction_csv='models/2023-12-22_ResNet101/predictions/predicted_age_fold-1.csv')
evaluator.generate_bland_altman_plot(databank_csv='/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti.csv',
                                     save_png='test.png',
                                     display_all_ages=True)

"""
'/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/dataset_splitting/spreadsheet/databank_dti.csv'
'diagnosis_simple'
    'normal': 'normal', 
    'Patient': 'patient', 
    'LMCI': 'MCI', 
    'impaired (not MCI)': 'patient',
    'No Cognitive Impairment': 'normal', 
    'dementia': 'dementia', 
    "Alzheimer's Dementia": 'dementia', 
    'CN': 'normal', 
    'MCI': 'MCI', 
    'EMCI': 'MCI', 
    'AD': 'dementia', 
    'Mild Cognitive Impairment': 'MCI', 
    'SMC': 'patient',
"""