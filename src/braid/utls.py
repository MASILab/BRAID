import subprocess
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_qa_screenshot_fa_md(path_fa, path_md, path_png, offset=0):
    
    list_dict_plot = [
        {
            'idx_row': 0,
            'path' : path_fa,
            'metric' : 'FA',
            'vmax' : 1,
            'vmin': 0
        },
        {
            'idx_row': 1,
            'path' : path_md,
            'metric' : 'MD',
            'vmax' : 0.003,
            'vmin': 0           
        }
    ]
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24, 14))
    
    for dict_plot in list_dict_plot:
        img = nib.load(dict_plot['path'])
        data = np.squeeze(img.get_fdata())
        resolution = data.shape[:3]
        
        # axial
        axes[dict_plot['idx_row'],0].set_title(dict_plot['metric'], fontsize=20)
        aspect = img.header.get_zooms()[1] / img.header.get_zooms()[0]
        axes[dict_plot['idx_row'],0].imshow(
            data[:,:,round(resolution[2]/2)+offset].T,
            cmap='gray',
            origin='lower',
            aspect=aspect,
            interpolation='nearest',
            vmin=dict_plot['vmin'],
            vmax=dict_plot['vmax']
        )
        
        # coronal
        aspect = img.header.get_zooms()[2] / img.header.get_zooms()[0]
        axes[dict_plot['idx_row'],1].imshow(
            data[:,round(resolution[1]/2)+offset,:].T,
            cmap='gray',
            origin='lower',
            aspect=aspect,
            interpolation='nearest',
            vmin=dict_plot['vmin'],
            vmax=dict_plot['vmax']
        )
            
        # sagittal
        aspect = img.header.get_zooms()[2] / img.header.get_zooms()[1]
        axes[dict_plot['idx_row'],2].imshow(
            data[round(resolution[0]/2)+offset+5,:,:].T,
            cmap='gray',
            origin='lower',
            aspect=aspect,
            interpolation='nearest',
            vmin=dict_plot['vmin'],
            vmax=dict_plot['vmax']
        )
    
    # Adjust the spacing and save the png
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    if Path(path_png).parent.exists() == False:
        subprocess.run(['mkdir', '-p', Path(path_png).parent])

    fig.savefig(path_png, bbox_inches='tight')
    plt.close('all')

def summarize_dataset(df):
    
    # generate unique subject ID across databank
    df['dataset_subject'] = df['dataset'] + '_' + df['subject']
    
    # for each subject, select the earliest scan with age available
    df_base = df.groupby('dataset_subject').apply(lambda x: x[x['age'].notnull()].nsmallest(1, 'age')).reset_index(drop=True)
    
    # cognitively normal
    num_subject = df_base.loc[df_base['control_label']==1, 'dataset_subject'].unique().shape[0]
    age_min = df_base.loc[df_base['control_label']==1, 'age'].min()
    age_max = df_base.loc[df_base['control_label']==1, 'age'].max()
    age_mean = df_base.loc[df_base['control_label']==1, 'age'].mean()
    age_std = df_base.loc[df_base['control_label']==1, 'age'].std()
    num_scans = df.loc[df['control_label']==1, ].shape[0]
    print(f"#Subjects (normal): {num_subject}\t"
          f"#Scans: {num_scans}\t"
          f"Age range: {age_min:.1f}-{age_max:.1f} yrs\t"
          f"mean ± std: {age_mean:.1f} ± {age_std:.1f} yrs")
                    
    # cognitively impaired
    num_subject = df_base.loc[df_base['control_label']==0, 'dataset_subject'].unique().shape[0]
    age_min = df_base.loc[df_base['control_label']==0, 'age'].min()
    age_max = df_base.loc[df_base['control_label']==0, 'age'].max()
    age_mean = df_base.loc[df_base['control_label']==0, 'age'].mean()
    age_std = df_base.loc[df_base['control_label']==0, 'age'].std()
    num_scans = df.loc[df['control_label']==0, ].shape[0]
    print(f"#Subjects (normal): {num_subject}\t"
          f"#Scans: {num_scans}\t"
          f"Age range: {age_min:.1f}-{age_max:.1f} yrs\t"
          f"mean ± std: {age_mean:.1f} ± {age_std:.1f} yrs")

    # total
    num_subject = df_base['dataset_subject'].unique().shape[0]
    age_min = df_base['age'].min()
    age_max = df_base['age'].max()
    age_mean = df_base['age'].mean()
    age_std = df_base['age'].std()
    num_scans = df.shape[0]
    print(f"#Subjects (normal): {num_subject}\t"
          f"#Scans: {num_scans}\t"
          f"Age range: {age_min:.1f}-{age_max:.1f} yrs\t"
          f"mean ± std: {age_mean:.1f} ± {age_std:.1f} yrs")
