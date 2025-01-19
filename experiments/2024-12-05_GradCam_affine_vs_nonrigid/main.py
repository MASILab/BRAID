# Run GradCAM.

import pdb
import yaml
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
from customization import load_trained_model, load_images
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
torch.set_flush_denormal(True)
device = torch.device('cuda')

dict_model = {
    'wm-age-nonrigid': {
        'dataset_rel_path': './braid_dataset_ss_affine_warp_crop_downsample',
        'suffix': '_skullstrip_MNI152_warped_crop_downsample.nii.gz',
    }, 
    'wm-age-affine': {
        'dataset_rel_path': 'braid_dataset_ss_affine_crop_downsample',
        'suffix': '_skullstrip_MNI152_crop_downsample.nii.gz',
    },
}

lut_sex = {
    'female': 0,
    'male': 1,
    'unknown': None,
}

lut_race = {
    'White': 1, 
    'Asian': 2, 
    'Black or African American': 3,
    'American Indian or Alaska Native': 4,
    'Some Other Race': 0,
    'unknown': 0,
}


if __name__ == '__main__':

    root_data = Path('/tmp/.GoneAfterReboot')
    root_weights = Path('/home-local/gaoc11/braid_go_public/braid-v1.0')
    df = pd.read_csv('/tmp/.GoneAfterReboot/braid_test.csv')

    for sex_label in df['sex']:
        for race_label in df['race_simple']:

            subset = df.loc[(df['sex']==sex_label) & (df['race_simple']==race_label), ]
            print(f"sex: {sex_label}\trace: {race_label}\t #samples: {len(subset.index)}")

            for model_type in dict_model.keys():
                path_yaml = root_weights / model_type / 'config.yaml'
                with open(path_yaml, 'r') as f:
                    config = yaml.safe_load(f)
                
                for fold in [1,2]:
                    model = load_trained_model(
                        model_name = config['model']['name'],
                        mlp_hidden_layer_sizes = config['model']['mlp_hidden_layer_sizes'],
                        feature_vector_length = config['model']['feature_vector_length'],
                        n_input_channels = config['model']['n_input_channels'],
                        sex = lut_sex[sex_label],
                        race = lut_race[race_label],
                        path_pth = str(root_weights / model_type / f'{model_type}-fold-{fold}.pth'), 
                        device = device,
                    )
                    model.eval()

                    target_layers = [model.layer4[-1]]
                    cam = GradCAM(model=model, target_layers=target_layers)
                    targets = [BinaryClassifierOutputTarget(1)]

                    for _, row in tqdm(subset.iterrows(), total=len(subset.index), desc=f"{model_type} fold-{fold}"):
                        path_fa = root_data / dict_model[model_type]['dataset_rel_path'] / row['dataset'] / row['subject'] / row['session'] / f"scan-{row['scan']}" / f"fa{dict_model[model_type]['suffix']}"
                        path_md = root_data / dict_model[model_type]['dataset_rel_path'] / row['dataset'] / row['subject'] / row['session'] / f"scan-{row['scan']}" / f"md{dict_model[model_type]['suffix']}"
                        img = load_images(path_fa, path_md)
                        img = img.to(device)

                        grayscale_cam = cam(input_tensor=img, targets=targets)
                        grayscale_cam = grayscale_cam[0,:].astype(np.float16)

                        # save to nifti
                        img_ref = nib.load(path_fa)
                        img_save = nib.Nifti1Image(grayscale_cam, affine=img_ref.affine, header=img_ref.header)
                        path_save = path_fa.parent / f"gradcam_fold-{fold}.nii.gz"
                        nib.save(img_save, path_save)
                        