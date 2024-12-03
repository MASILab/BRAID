import json
import hashlib
import subprocess
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path

def calculate_md5(file_path):
    """Calculate the MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def verify_downloaded_model_weights(root_weights):
    """Verify the downloaded model weights by comparing the MD5 hash."""
    root_weights = Path(root_weights)
    md5_json = root_weights / 'md5.json'
    
    if not md5_json.is_file():
        raise ValueError(f"md5.json not found in {root_weights}. Provide the path to the repository pulled from Hugging Face.")
        
    with open(md5_json, "r") as f:
        dict_md5 = json.load(f)

    for model_type in dict_md5.keys():
        for model_fn, md5 in dict_md5[model_type].items():
            model_path = root_weights / model_type / model_fn
            if not model_path.is_file():
                print(f"Model weights not found: {model_path}")
                continue
            
            md5_calculated = calculate_md5(model_path)
            if md5_calculated != md5:
                raise ValueError(f"MD5 mismatch for {model_path}! Expected: {md5}, Calculated: {md5_calculated}")
    print(f"MD5 match for all model weights in {root_weights}!")
    

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
    print(f"#Subjects (impair): {num_subject}\t"
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
    print(f"#Subjects (total ): {num_subject}\t"
          f"#Scans: {num_scans}\t"
          f"Age range: {age_min:.1f}-{age_max:.1f} yrs\t"
          f"mean ± std: {age_mean:.1f} ± {age_std:.1f} yrs")


def generate_png_during_training(img_tensor, path_png):
    """function for generating PNG for sanity check during model training.

    Args:
        img_tensor (torch.tensor): torch tensor (on CPU) in (2, 128, 152, 128) or (1, 128, 152, 128)
        path_png (str): path to save the PNG 
    """

    if img_tensor.shape[0] == 2:
        # DTI scalars
        list_dict_plot = [
            {'idx': 0, 'metric' : 'FA', 'vmax' : 1, 'vmin': 0},
            {'idx': 1, 'metric' : 'MD', 'vmax' : 0.003, 'vmin': 0},
            ]
        
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24, 14))
    
        for dict_plot in list_dict_plot:
            data = img_tensor[dict_plot['idx'],:,:,:].numpy()
            resolution = data.shape[:3]
            
            # axial
            axes[dict_plot['idx'], 0].set_title(dict_plot['metric'], fontsize=20)
            axes[dict_plot['idx'], 0].imshow(
                data[:,:,round(resolution[2]/2)].T,
                cmap='gray',
                origin='lower',
                interpolation='nearest',
                vmin=dict_plot['vmin'],
                vmax=dict_plot['vmax']
            )
            
            # coronal
            axes[dict_plot['idx'], 1].imshow(
                data[:,round(resolution[1]/2),:].T,
                cmap='gray',
                origin='lower',
                interpolation='nearest',
                vmin=dict_plot['vmin'],
                vmax=dict_plot['vmax']
            )
                
            # sagittal
            axes[dict_plot['idx'], 2].imshow(
                data[round(resolution[0]/2)+5,:,:].T,
                cmap='gray',
                origin='lower',
                interpolation='nearest',
                vmin=dict_plot['vmin'],
                vmax=dict_plot['vmax']
            )
    elif img_tensor.shape[0] == 1:
        # T1w
        data = img_tensor[0,:,:,:].numpy()
        resolution = data.shape[:3]
        vmin = data.min()
        vmax = data.max()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(22, 7))
        
        # axial
        im = axes[0].imshow(
            data[:,:,round(resolution[2]/2)].T,
            cmap='gray',
            origin='lower',
            interpolation='nearest',
            norm=norm
        )
                
        # coronal
        axes[1].imshow(
            data[:,round(resolution[1]/2),:].T,
            cmap='gray',
            origin='lower',
            interpolation='nearest',
            norm=norm
        )
                
        # sagittal
        axes[2].imshow(
            data[round(resolution[0]/2)+5,:,:].T,
            cmap='gray',
            origin='lower',
            interpolation='nearest',
            norm=norm
        )
        
        # Colorbar
        cbar_ax = fig.add_axes([0.65, 0.1, 0.25, 0.025]) # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        
    # Adjust the spacing and save the png
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    if Path(path_png).parent.exists() == False:
        subprocess.run(['mkdir', '-p', Path(path_png).parent])

    fig.savefig(path_png, bbox_inches='tight')
    plt.close('all')


def generate_qa_screenshot_t1w(path_t1w, path_png, offset=0):
    
    # load data and min-max normalization
    img = nib.load(path_t1w)
    data = np.squeeze(img.get_fdata())
    resolution = data.shape[:3]
    vmin = data.min()
    vmax = data.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(22, 7))
    
    # axial
    aspect = img.header.get_zooms()[1] / img.header.get_zooms()[0]
    im = axes[0].imshow(
        data[:,:,round(resolution[2]/2)+offset].T,
        cmap='gray',
        origin='lower',
        aspect=aspect,
        interpolation='nearest',
        norm=norm
    )
            
    # coronal
    aspect = img.header.get_zooms()[2] / img.header.get_zooms()[0]
    axes[1].imshow(
        data[:,round(resolution[1]/2)+offset,:].T,
        cmap='gray',
        origin='lower',
        aspect=aspect,
        interpolation='nearest',
        norm=norm
    )
            
    # sagittal
    aspect = img.header.get_zooms()[2] / img.header.get_zooms()[1]
    axes[2].imshow(
        data[round(resolution[0]/2)+offset+5,:,:].T,
        cmap='gray',
        origin='lower',
        aspect=aspect,
        interpolation='nearest',
        norm=norm
    )
    
    # Colorbar
    cbar_ax = fig.add_axes([0.65, 0.1, 0.25, 0.025]) # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    
    # Adjust the spacing and save the png
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    if Path(path_png).parent.exists() == False:
        subprocess.run(['mkdir', '-p', Path(path_png).parent])

    fig.savefig(path_png, bbox_inches='tight')
    plt.close('all')
