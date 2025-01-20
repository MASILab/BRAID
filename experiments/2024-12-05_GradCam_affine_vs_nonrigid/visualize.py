"""
I am thinking of a figure in this layout:

                  45-60    60-75   75-90
model A
model B

"""

import pdb
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from main import dict_model

root_data = Path('/tmp/.GoneAfterReboot')
outdir = Path('experiments/2024-12-05_GradCam_affine_vs_nonrigid/data/averaged_gradcam_fold-1')
df = pd.read_csv('/tmp/.GoneAfterReboot/braid_test.csv')

def compute_averaged_gradcam():

    for age_min, age_max in [(45,60), (60,75), (75,90)]:
        subset = df.loc[df['age'].between(age_min, age_max), ]
        
        for model_type in dict_model.keys():
            for layer in [1,2,3,4]:

                existing_gradcam_images = []
                for _, row in subset.iterrows():
                    path_gradcam = root_data / dict_model[model_type]['dataset_rel_path'] / row['dataset'] / row['subject'] / row['session'] / f"scan-{row['scan']}" / f"gradcam_fold-1_layer{layer}.nii.gz"
                    if path_gradcam.exists():
                        existing_gradcam_images.append(path_gradcam)
                
                if len(existing_gradcam_images) == 0:
                    raise ValueError(f"There is not a single GradCAM nifti for {age_min}-{age_max}_{model_type}_layer{layer}")

                for i, path_gradcam in tqdm(enumerate(existing_gradcam_images), total=len(existing_gradcam_images), desc=f"{age_min}-{age_max}_{model_type}_layer{layer}"):
                    img_gradcam = nib.load(path_gradcam)
                    data_gradcam = img_gradcam.get_fdata()
                    if i==0:
                        data_averaged = data_gradcam / len(existing_gradcam_images)
                    else:
                        data_averaged += data_gradcam / len(existing_gradcam_images)

                # Save to nifti
                path_save = outdir / f"{age_min}-{age_max}_{model_type}_layer{layer}.nii.gz"
                img_save = nib.Nifti1Image(data_averaged, affine=img_gradcam.affine, header=img_gradcam.header)
                nib.save(img_save, path_save)
                print(f"Averaged GradCAM saved to: {path_save}")


def make_figure():
    figs_dir = Path('experiments/2024-12-05_GradCam_affine_vs_nonrigid/figs')

    img_mni = nib.load('experiments/2024-12-05_GradCam_affine_vs_nonrigid/data/MNI_152_crop_downsample.nii.gz')
    data_mni = img_mni.get_fdata()
    aspect = img_mni.header.get_zooms()[1] / img_mni.header.get_zooms()[0]

    for layer in [1,2,3,4]:
        for idx_axial in range(40, 70, 3):
            
            # make figure for each layer and each axial position
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(5,4))

            for idx_row, model_type in enumerate(['wm-age-affine', 'wm-age-nonrigid']):
                for idx_col, age_range in enumerate([(45,60), (60,75), (75,90)]):
                    
                    path_img = outdir / f"{age_range[0]}-{age_range[1]}_{model_type}_layer{layer}.nii.gz"
                    img = nib.load(path_img)
                    data = img.get_fdata()

                    axes[idx_row, idx_col].imshow(
                        data_mni[:,:,idx_axial].T,
                        cmap='gray',
                        origin='lower',
                        aspect=aspect,
                        interpolation='nearest',
                        )
                    
                    axes[idx_row, idx_col].imshow(
                        data[:,:,idx_axial].T,
                        cmap='viridis',
                        origin='lower',
                        aspect=aspect,
                        interpolation='bilinear',
                        alpha=0.5,
                        )
                    
                    axes[idx_row, idx_col].set_xticks([])
                    axes[idx_row, idx_col].set_yticks([])
        
            plt.subplots_adjust(wspace=0.02, hspace=0.02)
            png = figs_dir / f'layer{layer}' / f'{idx_axial}.png'
            fig.savefig(png, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close('all')

if __name__ == '__main__':
    compute_averaged_gradcam()
    make_figure()

