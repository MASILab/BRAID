# Visualize the distribution of the data in the data bank.
# Author: Chenyu Gao
# Date: Nov 13, 2023

from pathlib import Path
import pandas as pd
import seaborn as sns

clean_csv_root = Path('./data/subject_info/clean')

dict_dx2standard = {
    'No Cognitive Impairment': 'normal',
    'Mild Cognitive Impairment': 'MCI',
    "Alzheimer's Dementia": 'AD',
    'CN': 'normal',
    'EMCI': 'other_disease',
    'SMC': 'other_disease',
    'LMCI': 'other_disease',
    'Patient': 'other_disease',
    }

for csv in clean_csv_root.iterdir():
    if not csv.name.endswith('_info.csv'):
        continue
    
    # Load the csv for each dataset
    dataset_name = csv.name.split('_')[0]
    demog = pd.read_csv(csv)

    if dataset_name == 'UKBB':
        # use the CNS_control_2 column to determine class
        demog['diagnosis'] = demog['CNS_control_2'].apply(lambda x: 'control' if x == 1 else 'disease')
        plot = sns.displot(demog, x="age", col="diagnosis", row="sex", hue='race', multiple='stack',
                           binwidth=3, height=3, facet_kws=dict(margin_titles=True))
    else:
        # remap diagnosis for visualizations
        print("{} diagnosis classes before remapping: ".format(dataset_name), demog['diagnosis'].unique())
        demog['diagnosis'] = demog['diagnosis'].map(dict_dx2standard).fillna(demog['diagnosis'])
        print("{} diagnosis classes after remapping: ".format(dataset_name), demog['diagnosis'].unique())

        plot = sns.displot(demog, x="age", col="diagnosis", row="sex", hue='race', multiple='stack',
                        binwidth=3, height=3, facet_kws=dict(margin_titles=True))
    
    plot.fig.suptitle(dataset_name, y=1.02)
    plot.savefig(f"./data/dataset_splitting/figs/displot/{dataset_name}_distribution.png")
    