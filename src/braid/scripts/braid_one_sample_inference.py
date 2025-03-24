import os
import yaml
import json
import torch
import shutil
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import braid.calculate_dti_scalars
import braid.utls
import braid.evaluations
import braid.registrations
from pathlib import Path
from monai.transforms import (
    Compose,
    LoadImage,
    LoadImaged,
    EnsureChannelFirst,
    EnsureChannelFirstd,
    Orientation,
    Orientationd,
    CenterSpatialCrop,
    CenterSpatialCropd,
    Spacing,
    Spacingd,
    ToTensor,
    ToTensord,
    ConcatItemsd,
    NormalizeIntensity
)
torch.set_flush_denormal(True)


def convert_to_brain_mask(path_input, path_output, background=0):
    """Convert the segmentation image to a binary brain mask.

    Args:
        path_input (str): path to the segmentation image.
        path_output (str): path to the output brain mask.
    """
    img = nib.load(path_input)
    data = img.get_fdata()
    brain_mask = (data != background).astype("uint8")
    nib.save(nib.Nifti1Image(brain_mask, img.affine), path_output)


def vectorize_label(label_name, label_value):
    
    if label_name == 'sex':
        lut = {
            0: [1, 0],
            1: [0, 1],
        }
        if label_value in lut.keys():
            return lut[label_value]
        else:
            return [0.5, 0.5]

    elif label_name == 'race':
        lut = {
            1: [1, 0, 0, 0], 
            2: [0, 1, 0, 0], 
            3: [0, 0, 1, 0],
            4: [0, 0, 0, 1],
            0: [0.25, 0.25, 0.25, 0.25],
        }
        if label_value in lut.keys():
            return lut[label_value]
        else:
            return [0.25, 0.25, 0.25, 0.25]
    
    else:
        raise ValueError(f'Unknown label name: {label_name}')


def load_braid_sample(modality, path_fa=None, path_md=None, path_t1=None, sex=None, race=None):
    """Load and preprocess a single sample.

    Args:
        modality (str): Modality of the input image(s). It can be 'DTI' or 'T1w'.
        path_fa (str | PosixPath): Path to the FA image in the MNI152 space.
        path_md (str | PosixPath): Path to the MD image in the MNI152 space.
        path_t1 (str | PosixPath): Path to the T1w image in the MNI152 space.
        sex (int): sex provided by user.
        race (int): race provided by user.

    Returns:
        torch.Tensor: A tensor of shape (1, C, D, H, W) containing the image(s) with a batch dimension.
        torch.Tensor: A tensor of shape (1, 6) containing the label feature vector with a batch dimension.    
    """
    
    if modality == 'DTI':
        transform = Compose([
            LoadImaged(keys=['fa', 'md'], image_only=False),
            EnsureChannelFirstd(keys=['fa', 'md']),
            Orientationd(keys=['fa', 'md'], axcodes="RAS"),
            CenterSpatialCropd(keys=['fa', 'md'], roi_size=(192, 228, 192)),
            Spacingd(keys=['fa', 'md'], pixdim=(1.5, 1.5, 1.5), mode=('bilinear', 'bilinear')),
            ToTensord(keys=['fa', 'md']),
            ConcatItemsd(keys=['fa', 'md'], name='images')
        ])
        data_dict = {'fa': path_fa, 'md': path_md}
        data_dict = transform(data_dict)
        img = data_dict['images']
    
    elif modality == 'T1w':
        transform = Compose([
            LoadImage(reader="NibabelReader", image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            CenterSpatialCrop(roi_size=(192, 228, 192)),
            Spacing(pixdim=(1.5, 1.5, 1.5), mode='bilinear'),
            NormalizeIntensity(nonzero=True),
            ToTensor(),
        ])
        img = transform(path_t1)
    else:
        raise ValueError(f'Unknown modality: {modality}')
    
    img = img.unsqueeze(0)
        
    sex_vec = vectorize_label(label_name='sex', label_value=sex)
    sex_vec = torch.tensor(sex_vec, dtype=torch.float32)
    race_vec = vectorize_label(label_name='race', label_value=race)
    race_vec = torch.tensor(race_vec, dtype=torch.float32)
    label_feature_vec = torch.cat((sex_vec, race_vec), dim=0)
    label_feature_vec = label_feature_vec.unsqueeze(0)
    
    return img, label_feature_vec


def generate_qa_png(mni152, fa_warp, md_warp, fa_affine, md_affine, t1w_affine, csv, png):
    df = pd.read_csv(csv)
    brain_ages = ['wm-age-nonrigid', 'wm-age-affine', 'gm-age-ours']
    brain_ages_rename = ['WM age (nonrigid)', 'WM age (affine)', 'GM age (affine/ours)']
    path_imgs = [[fa_warp, md_warp], [fa_affine, md_affine], [t1w_affine]]
    ranges = [[(0, 1), (0, 0.003)], [(0, 1), (0, 0.003)], [(None, None)]]
    offsets = [-20, -10, 0]
    
    fontsize = 8
    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(nrows=6, ncols=6, hspace=0, wspace=0.02, height_ratios=[1,1,1,6,6,6])
    
    # Text
    txt = ['Brain age:', 'uncorrected', 'bias-corrected']
    for i, t in enumerate(txt):
        ax = fig.add_subplot(gs[i,0])
        ax.text(0.5, 0.5, t, ha='center', va='center', fontsize=fontsize)
        ax.axis('off')
    
    # MNI152
    img = nib.load(mni152)
    data = img.get_fdata()
    resolution = data.shape[:3]
    aspect = img.header.get_zooms()[1] / img.header.get_zooms()[0]
    for i, offset in enumerate(offsets):
        ax = fig.add_subplot(gs[3+i, 0])
        ax.imshow(
            data[:,:,int(resolution[2]/2)+offset].T,
            cmap='gray',
            origin='lower',
            aspect=aspect,
            interpolation='nearest'
            )
        ax.axis('off')
    
    # Input images
    for i, brain_age in enumerate(brain_ages):
        paths = path_imgs[i]
        
        # sub-title
        ax = fig.add_subplot(gs[0, i*2+1:i*2+1+len(paths)])
        ax.text(0.5, 0.5, brain_ages_rename[i], ha='center', va='center', fontsize=fontsize)
        ax.axis('off')
        
        # uncorrected value
        mean = df[f'{brain_age}_mean'].values[0]
        std = df[f'{brain_age}_std'].values[0]
        txt = f'{mean:.1f} ± {std:.1f}' if not np.isnan(mean) else 'N/A'
        ax = fig.add_subplot(gs[1, i*2+1:i*2+1+len(paths)])
        ax.text(0.5, 0.5, txt, ha='center', va='center', fontsize=fontsize)
        ax.axis('off')
        
        # bias-corrected value
        mean = df[f'{brain_age}_bias-corrected_mean'].values[0]
        std = df[f'{brain_age}_bias-corrected_std'].values[0]
        txt = f'{mean:.1f} ± {std:.1f}' if not np.isnan(mean) else 'N/A'
        ax = fig.add_subplot(gs[2, i*2+1:i*2+1+len(paths)])
        ax.text(0.5, 0.5, txt, ha='center', va='center', fontsize=fontsize)
        ax.axis('off')
        
        # images
        for j, path_img in enumerate(paths):
            idx_col = i*2 + 1 + j
            
            img = nib.load(path_img)
            data = img.get_fdata()
            resolution = data.shape[:3]
            aspect = img.header.get_zooms()[1] / img.header.get_zooms()[0]
            
            for k, offset in enumerate(offsets):
                idx_row = 3 + k
                ax = fig.add_subplot(gs[idx_row, idx_col])
                ax.imshow(
                    data[:, :, int(resolution[2]/2)+offset].T,
                    cmap='gray',
                    origin='lower',
                    aspect=aspect,
                    interpolation='nearest',
                    vmin=ranges[i][j][0],
                    vmax=ranges[i][j][1],
                )
                ax.axis('off')
        
    fig.savefig(png, dpi=300, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(
        prog='BRAID inference for one sample',
        description=(
            'Run BRAID model inference for one sample.' 
            'It performs the following steps in sequence: '
            '1) Calculation of FA and MD images; '
            '2) Registrations, between T1 and template, and between b0 and T1; '
            '3) Transform the FA and MD images to the template space (affine or non-rigid); '
            '4) Run five independently trained ResNet models obtained through five-fold cross-validation, '
            'for WM age nonrigid, WM age affine, and GM age ours, respectively. '
            'In the end, for each brain age estimation type, '
            'it provides five individual estimations, mean, and '
            'standard deviation of the estimations in a single-row csv file.'
        )
    )
    parser.add_argument('-d', '--dwi', type=str, required=True, help='path to the NIFTI file of the diffusion-weighted MR image.')
    parser.add_argument('-v', '--bval', type=str, required=True, help='path to the b value table (typically a file ending with ".bval") for the diffusion-weighted MR image.')
    parser.add_argument('-c', '--bvec', type=str, required=True, help='path to the b vector table (typically a file ending with "*.bvec") for the diffusion-weighted MR image.')
    parser.add_argument('-t', '--t1w', type=str, required=True, help='path to the NIFTI file of the T1-weighted MR image of the same subject, preferably acquired in the same imaging session as the diffusion-weighted MR image.')
    parser.add_argument('-tm', '--t1w_mask', type=str, required=True, help='path to the NIFTI file of the brain mask for the T1-weighted MR image. Segmentation image is also acceptable if the background is labeled as 0 and the brain is labeled with other integer values than 0.')
    parser.add_argument('-m', '--mni152', type=str, required=True, help='path to the NIFTI file of the MNI152 template image, which is provided in data/template/MNI_152.nii.gz.')
    parser.add_argument('-a', '--age', type=float, required=False, help='(optional, but recommended) A float representing chronological age in years. This is used to correct a common bias in brain age estimation, where older subjects tend to be underestimated and younger subjects overestimated.')
    parser.add_argument('-s', '--sex', type=int, required=False, help='(optional, but recommended) integer, 0 for female, 1 for male.')
    parser.add_argument('-r', '--race', type=int, required=False, help='(optional, but recommended) integer, 1 for "White", 2 for "Asian", 3 for "Black or African American", 4 for "American Indian or Alaska Native", and 0 for "Some Other Race".')
    parser.add_argument('-w', '--weights', type=str, required=True, help='path to the directory of model weights pulled from Hugging Face. For example, /home-local/inference/braid-v1.0.')
    parser.add_argument('-cc', '--check_complete', action='store_true', help='if the flag is given, the code will check the completeness of the model weights repository and verify the MD5 hash of the .pth files.')
    parser.add_argument('-i', '--intermediate', action='store_true', help='if the flag is given, the intermediate processing files will be preserved after the job completion. It is recommended to check the intermediate files, especially the processed images, before analysis.')
    parser.add_argument('-po', '--preprocess_only', action='store_true', help='if the flag is given, it will run everything except for the ResNet part. This is useful when the user want to run the preprocessing in parallel on a CPU-only machine (which may have more cores) first, and then switch to a GPU machine to complete the inference.')
    parser.add_argument('-co', '--cpu_only', action='store_true', help='if the flag is given, the code will run on CPU only.')
    parser.add_argument('-g', '--gpu', type=int, required=False, help='(optional) integer, the index of the GPU to use. If not provided (and not in cpu-only mode), the code will use the first available GPU.')
    parser.add_argument('-o', '--outdir', type=str, required=True, help='path to the output directory (write permission required).')
    args = parser.parse_args()

    print("Start BRAID inference for one sample...")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # step 1: brain mask and skull-stripping for T1w image
    path_t1w_brain_mask = outdir / 't1w_brain_mask.nii.gz'
    if path_t1w_brain_mask.is_file():
        print(f'Brain mask for T1w image already exists at {path_t1w_brain_mask}')
    else:
        convert_to_brain_mask(args.t1w_mask, path_t1w_brain_mask)
        print(f'Saved brain mask for T1w image at {path_t1w_brain_mask}')

    path_t1w_brain = outdir / 't1w_brain.nii.gz'
    if path_t1w_brain.is_file():
        print(f'Skull-stripped T1w image already exists at {path_t1w_brain}')
    else:
        braid.calculate_dti_scalars.apply_skull_strip_mask(args.t1w, path_t1w_brain_mask, path_t1w_brain)
        print(f'Saved brain image at {path_t1w_brain}')

    # step 2: extract single shell and b0
    path_dwi_out = outdir / 'dwmri_firstshell.nii.gz'
    path_bval_out = outdir / 'dwmri_firstshell.bval'
    path_bvec_out = outdir / 'dwmri_firstshell.bvec'
    path_b0_out = outdir / 'dwmri_firstshell_b0.nii.gz'
    if path_dwi_out.is_file() and path_bval_out.is_file() and path_bvec_out.is_file() and path_b0_out.is_file():
        print(f'Single shell and b0 images already exist at {path_dwi_out.parent}')
    else:
        braid.calculate_dti_scalars.extract_single_shell(args.dwi, args.bval, args.bvec, path_dwi_out, path_bval_out, path_bvec_out, threshold=1500)
        braid.calculate_dti_scalars.extract_b0_volume(path_dwi_out, path_bval_out, path_bvec_out, path_b0_out)
        print(f'Extracted single shell and b0 images at {path_dwi_out.parent}')

    # step 3: registrations (b0 to t1, t1 to template affine, t1 to template non-rigid)
    transform_b0_to_t1 = outdir / 'transform_b0tot1.txt'
    transform_t1_to_b0 = outdir / 'transform_t1tob0.txt'
    transform_t1_to_template_affine = outdir / 'transform_t1toMNI_affine.mat'
    transform_t1_to_template_warp = outdir / 'transform_t1toMNI_warp.nii.gz'
    if transform_b0_to_t1.is_file() and transform_t1_to_b0.is_file() and transform_t1_to_template_affine.is_file() and transform_t1_to_template_warp.is_file():
        print(f'Registration files already exist at {transform_b0_to_t1.parent}')
    else:
        braid.registrations.register_b0_to_MNI152(path_b0_out, args.t1w, path_t1w_brain, args.mni152, outdir)
        print(f'Registration files saved at {transform_b0_to_t1.parent}')

    # step 4: calculate FA and MD maps
    path_b0_brain_mask = outdir / 'brain_mask_b0.nii.gz'
    path_b0_brain_mask_dilated = outdir / 'brain_mask_b0_dilated.nii.gz'
    path_tensor = outdir / 'tensor.nii.gz'
    path_fa = outdir / 'fa.nii.gz'
    path_md = outdir / 'md.nii.gz'
    if path_fa.is_file() and path_md.is_file() and path_tensor.is_file() and path_b0_brain_mask.is_file() and path_b0_brain_mask_dilated.is_file():
        print(f'FA and MD maps already exist at {path_fa.parent}')
    else:
        braid.calculate_dti_scalars.calculate_fa_md_maps(path_dwi_out, path_bval_out, path_bvec_out, path_b0_out, path_t1w_brain_mask, transform_t1_to_b0, path_b0_brain_mask, path_b0_brain_mask_dilated, path_tensor, path_fa, path_md)
        print(f'FA and MD maps saved at {path_fa.parent}')
    
    # step 5: skull-strip FA and MD maps
    path_fa_ss = outdir / 'fa_skullstrip.nii.gz'
    path_md_ss = outdir / 'md_skullstrip.nii.gz'

    if path_fa_ss.is_file() and path_md_ss.is_file():
        print(f'Skull-stripped FA and MD maps already exist at {path_fa_ss.parent}')
    else:
        braid.calculate_dti_scalars.apply_skull_strip_mask(path_fa, path_b0_brain_mask, path_fa_ss)
        braid.calculate_dti_scalars.apply_skull_strip_mask(path_md, path_b0_brain_mask, path_md_ss)
        print(f'Skull-stripped FA and MD maps saved at {path_fa_ss.parent}')

    # step 6: transform FA and MD maps and T1w image to MNI152 space
    path_fa_ss_mni_affine = outdir / 'fa_skullstrip_MNI152.nii.gz'
    path_md_ss_mni_affine = outdir / 'md_skullstrip_MNI152.nii.gz'
    path_fa_ss_mni_warp = outdir / 'fa_skullstrip_MNI152_warped.nii.gz'
    path_md_ss_mni_warp = outdir / 'md_skullstrip_MNI152_warped.nii.gz'
    if path_fa_ss_mni_affine.is_file() and path_md_ss_mni_affine.is_file() and path_fa_ss_mni_warp.is_file() and path_md_ss_mni_warp.is_file():
        print(f'Transformed FA and MD maps already exist at {path_fa_ss_mni_affine.parent}')
    else:
        t_affine = [transform_t1_to_template_affine, transform_b0_to_t1]
        braid.registrations.apply_transform_to_img_in_b0(path_fa_ss, args.mni152, path_fa_ss_mni_affine, t_affine)
        braid.registrations.apply_transform_to_img_in_b0(path_md_ss, args.mni152, path_md_ss_mni_affine, t_affine)
        
        t_warp = [transform_t1_to_template_warp, transform_t1_to_template_affine, transform_b0_to_t1]
        braid.registrations.apply_transform_to_img_in_b0(path_fa_ss, args.mni152, path_fa_ss_mni_warp, t_warp)
        braid.registrations.apply_transform_to_img_in_b0(path_md_ss, args.mni152, path_md_ss_mni_warp, t_warp)
        print(f'Transformed FA and MD maps saved at {path_fa_ss_mni_affine.parent}')
    
    path_t1w_brain_mni_affine = outdir / 't1w_brain_MNI152.nii.gz'
    if path_t1w_brain_mni_affine.is_file():
        print(f'Transformed T1w image already exists at {path_t1w_brain_mni_affine}')
    else:
        braid.registrations.apply_ants_transformations(path_t1w_brain, args.mni152, path_t1w_brain_mni_affine, list_transforms=[transform_t1_to_template_affine])
        print(f'Transformed T1w image saved at {path_t1w_brain_mni_affine}')
    
    if args.preprocess_only:
        print('All preprocessing steps are completed.\n')
        return
    
    # step 7: run ResNet models
    root_weights = Path(args.weights)
    if args.check_complete:
        braid.utls.verify_downloaded_model_weights(root_weights)
    else:
        print('Skip verifying the model weights repository.')
    
    if args.cpu_only:
        device = torch.device('cpu')
    elif args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device} for model inference.\n')
    
    row_data = {'path_dwi': [args.dwi], 'path_t1w': [args.t1w]}
    for model_type in ['wm-age-nonrigid', 'wm-age-affine', 'gm-age-ours']:
        
        path_yaml = root_weights / model_type / 'config.yaml'
        with open(path_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        # load sample to device
        if model_type == 'wm-age-nonrigid':
            img, label_feature_vec = load_braid_sample(modality='DTI', path_fa=path_fa_ss_mni_warp, path_md=path_md_ss_mni_warp, sex=args.sex, race=args.race)
        elif model_type == 'wm-age-affine':
            img, label_feature_vec = load_braid_sample(modality='DTI', path_fa=path_fa_ss_mni_affine, path_md=path_md_ss_mni_affine, sex=args.sex, race=args.race)
        elif model_type == 'gm-age-ours':
            img, label_feature_vec = load_braid_sample(modality='T1w', path_t1=path_t1w_brain_mni_affine, sex=args.sex, race=args.race)
        else:
            raise ValueError(f'Unknown model type: {model_type}')
        img, label_feature_vec = img.to(device), label_feature_vec.to(device)
        
        for fold in [1,2,3,4,5]:
            # load model
            model = braid.evaluations.load_trained_model(
                model_name = config['model']['name'],
                mlp_hidden_layer_sizes = config['model']['mlp_hidden_layer_sizes'],
                feature_vector_length = config['model']['feature_vector_length'],
                n_input_channels = config['model']['n_input_channels'],
                path_pth = str(root_weights / model_type / f'{model_type}-fold-{fold}.pth'), 
                device = device
                )
            model.eval()
            with torch.no_grad():
                pred = model(img, label_feature_vec).detach().cpu()
                pred = torch.flatten(pred).tolist()[0]    
            
            # bias corrections
            pred_bc = None
            if args.age is None:
                print("Chronological age is not provided. Skip bias correction.")
            else:
                with open(str(root_weights / model_type / f'{model_type}-fold-{fold}-bc-params.json'), 'r') as f:
                    bc_params = json.load(f)
                pred_bc = pred - (bc_params['slope']*args.age + bc_params['intercept'])
            
            row_data[f'{model_type}_fold-{fold}'] = [pred]
            row_data[f'{model_type}_fold-{fold}_bias-corrected'] = [pred_bc]
            
        # Compute mean and std
        vals = []
        for fold in [1,2,3,4,5]:
            val = row_data[f'{model_type}_fold-{fold}'][0]
            if val:
                vals.append(val)
        if len(vals) > 0:
            row_data[f'{model_type}_mean'] = [np.nanmean(vals)]
            row_data[f'{model_type}_std'] = [np.nanstd(vals)]
        else:
            row_data[f'{model_type}_mean'] = [None]
            row_data[f'{model_type}_std'] = [None]
        
        vals = []
        for fold in [1,2,3,4,5]:
            val = row_data[f'{model_type}_fold-{fold}_bias-corrected'][0]
            if val:
                vals.append(val)
        if len(vals) > 0:
            row_data[f'{model_type}_bias-corrected_mean'] = [np.nanmean(vals)]
            row_data[f'{model_type}_bias-corrected_std'] = [np.nanstd(vals)]
        else:
            row_data[f'{model_type}_bias-corrected_mean'] = [None]
            row_data[f'{model_type}_bias-corrected_std'] = [None]
    
    df = pd.DataFrame(row_data)
    path_csv = outdir / 'final' / 'braid_predictions.csv'
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_csv, index=False)
    print(f'Brain age estimates saved at {path_csv}')
    
    # step 8: generate QA png
    png = outdir / 'final' / 'QA.png'
    generate_qa_png(
        mni152=args.mni152, 
        fa_warp=path_fa_ss_mni_warp, 
        md_warp=path_md_ss_mni_warp, 
        fa_affine=path_fa_ss_mni_affine, 
        md_affine=path_md_ss_mni_affine, 
        t1w_affine=path_t1w_brain_mni_affine, 
        csv=path_csv, 
        png=png,
        )
    
    # step 9: remove all intermediate files, if not specified to preserve
    if not args.intermediate:
        for fn in outdir.iterdir():
            if fn.name == 'final':
                continue
            if fn.is_dir():
                shutil.rmtree(fn)
            else:
                os.remove(fn)
        print('Intermediate files removed.')
