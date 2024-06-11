import re
import torch
import shutil
import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    CenterSpatialCrop,
    Spacing,
    SaveImage,
)
torch.set_flush_denormal(True)


class CropDownSampleDataset(Dataset):
    def __init__(self, list_nifti, data_root_dir):
        self.list_nifti = list_nifti
        self.transform = Compose([
            LoadImage(reader="NibabelReader", image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            CenterSpatialCrop(roi_size=(192, 228, 192)),
            Spacing(pixdim=(1.5, 1.5, 1.5), mode='bilinear'), # expected: 128 x 152 x 128, 1.5mm^3
            SaveImage(
                output_postfix='crop_downsample', 
                output_ext='.nii.gz', 
                output_dir=data_root_dir, 
                data_root_dir=data_root_dir,
                separate_folder=False,
                print_log=False,
                ),
        ])
        
    def __len__(self):
        return len(self.list_nifti)
    
    def __getitem__(self, idx):
        self.transform(self.list_nifti[idx])
        return 0


def prepare_braid_style_spreadsheet(output_csv='experiments/2024-06-06_VMAP_Model_Inference/vmap_braid_style.csv'):
    """ Prepare a spreadsheet of testing set in BRAID style.
    """
    demog = pd.read_csv('data/subject_info/raw/VMAP_2024-05_update/Landman_Diffusion_Harmonization_23-24_data_withdiagnosis.csv', index_col=0)    
    
    dict_sex2standard = {1: 'male', 2: 'female'}
    dict_race2standard = {
        1: 'White',
        2: 'Black or African American',
        3: 'American Indian or Alaska Native',
        4: 'Native Hawaiian or Other Pacific Islander',
        5: 'Asian',
        50: 'Some Other Race',
        99: None,    
    }

    dict_datasets = {
        'VMAP_JEFFERSON': 'JEFFERSON',
        'VMAP_2.0': 'JEFFERSONVMAP',
        'VMAP_TAP': 'JEFFERSONTAP'
    }

    data = {
        'dataset': [], 'subject': [], 'session': [], 'scan': [],
        'sex': [], 'race_simple': [], 'age': [], 'dataset_subject': [],  # above are required
        'path_wmatlas': [],
        'bids_acquisition_label': [], 'bids_run_label': [],
        'xnat_project': [], 'vmac_id': [], 'session_id': [],
    }

    for bids_dataset, xnat_project in dict_datasets.items():
        path_derivatives = Path('/nfs2/harmonization/BIDS') / bids_dataset / 'derivatives'
        subject_folders = [f for f in path_derivatives.iterdir() if f.is_dir() and f.name.startswith('sub-')]

        for subject in subject_folders:
            bids_subject_id = subject.name
            vmac_id = int(bids_subject_id.split('-')[1])
            session_folders = [f for f in subject.iterdir() if f.is_dir() and f.name.startswith('ses-')]

            for session in session_folders:
                bids_session_id = session.name
                session_id = bids_session_id.split('-')[1].split('x')[0]
                
                # collect subject information
                match = demog.loc[(demog['vmac_id'] == vmac_id)&(demog['session_id']==session_id), ]
                if len(match.index) == 1:
                    age = match['age'].values[0]
                    sex = match['sex'].values[0]
                    race = match['race'].values[0]
                elif len(match.index) == 0:
                    match = demog.loc[(demog['vmac_id'] == vmac_id)&(demog['session_id']==session_id[:6]), ]
                    if len(match.index) != 0:
                        age = match['age'].values[0]
                        sex = match['sex'].values[0]
                        race = match['race'].values[0]
                    else:
                        print(f'No match for {bids_subject_id} {bids_session_id}')
                        age = None
                        sex = None
                        race = None
                else:
                    raise ValueError(f'Multiple matches for {bids_subject_id} {bids_session_id}')
                
                sex = dict_sex2standard[sex] if sex in dict_sex2standard.keys() else None
                race = dict_race2standard[race] if race in dict_race2standard.keys() else None
                
                for i, wmatlas in enumerate(session.glob('./WMAtlas*')):
                    if 'EVE3' in wmatlas.name: continue
                    acq_match = re.search(r'acq-(\w+)', wmatlas.name)
                    run_match = re.search(r'run-(\d{1,2})', wmatlas.name)
                    bids_acquisition_label = acq_match.group(0) if acq_match else ""
                    bids_run_label = run_match.group(0) if run_match else ""                                        

                    # check if the wmatlas processing is complete
                    fa = wmatlas / 'dwmri%fa.nii.gz'
                    md = wmatlas / 'dwmri%md.nii.gz'
                    b0_mask = wmatlas / 'dwmri%_dwimask.nii.gz'
                    transform_b0_to_t1 = wmatlas / 'dwmri%ANTS_b0tot1.txt'
                    transform_t1_to_MNI_affine = wmatlas / 'dwmri%0GenericAffine.mat'
                    transform_t1_to_MNI_warp = wmatlas / 'dwmri%1Warp.nii.gz'
                    list_files_to_check = [
                        fa, md, b0_mask, 
                        transform_b0_to_t1, transform_t1_to_MNI_affine, transform_t1_to_MNI_warp
                    ]
                    if any(not file.exists() for file in list_files_to_check):
                        print(wmatlas, ' has missing files.')
                        continue
                    
                    data['dataset'].append(bids_dataset)
                    data['subject'].append(bids_subject_id)
                    data['session'].append(bids_session_id)
                    data['scan'].append(i+1)
                    data['sex'].append(sex)
                    data['race_simple'].append(race)
                    data['age'].append(age)
                    data['dataset_subject'].append(f'{bids_dataset}_{bids_subject_id}')
                    data['path_wmatlas'].append(str(wmatlas))
                    data['bids_acquisition_label'].append(bids_acquisition_label)
                    data['bids_run_label'].append(bids_run_label)
                    data['xnat_project'].append(xnat_project)
                    data['vmac_id'].append(vmac_id)
                    data['session_id'].append(session_id)
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

def apply_transform_to_img_in_b0(path_img_b0, path_img_ref, path_img_out, list_transforms):
    """
    Apply transformations to image in b0 space. This will align the image to the MNI152 standard template.
    path_img_b0: path to the image in b0 space
    path_img_ref: path to the reference image (MNI152 template image)
    path_img_out: path to the output image
    list_transforms: list of paths to the transformations
    """
    if not Path(path_img_out).parent.is_dir():
        subprocess.run(['mkdir', '-p', str(Path(path_img_out).parent)])
    command = [
        'antsApplyTransforms', '-d', '3',
        '-i', str(path_img_b0),
        '-r', str(path_img_ref), 
        '-o', str(path_img_out), 
        '-n', 'Linear', 
        ]
    for transform in list_transforms:
        command += ['-t', str(transform)]
        
    subprocess.run(command)

def get_skull_stripped_fa_md_in_mni152(job_tuple):
    path_wmatlas, output_dir, dataset, subject, session, scan, transformation = job_tuple

    wmatlas = Path(path_wmatlas)
    output_dir = Path(output_dir)
    path_MNI152 = '/nfs2/ForChenyu/MNI_152.nii.gz'

    # transfer required inputs to local disk    
    fa = wmatlas / 'dwmri%fa.nii.gz'
    md = wmatlas / 'dwmri%md.nii.gz'
    b0_mask = wmatlas / 'dwmri%_dwimask.nii.gz'
    transform_b0_to_t1 = wmatlas / 'dwmri%ANTS_b0tot1.txt'
    transform_t1_to_MNI_affine = wmatlas / 'dwmri%0GenericAffine.mat'
    transform_t1_to_MNI_warp = wmatlas / 'dwmri%1Warp.nii.gz'
    
    t_folder = output_dir / dataset / subject / session / f'scan-{scan}' / 'tmp'
    t_folder.mkdir(parents=True, exist_ok=True)

    t_fa = t_folder / 'fa.nii.gz'
    t_md = t_folder / 'md.nii.gz'
    t_b0_mask = t_folder / 'brain_mask_b0.nii.gz'
    t_transform_b0_to_t1 = t_folder / 'transform_b0tot1.txt'
    t_transform_t1_to_MNI_affine = t_folder / 'transform_t1toMNI_affine.mat'
    t_transform_t1_to_MNI_warp = t_folder / 'transform_t1toMNI_warp.nii.gz'

    subprocess.run(['rsync', '-L', str(fa), str(t_fa)])
    subprocess.run(['rsync', '-L', str(md), str(t_md)])
    subprocess.run(['rsync', '-L', str(b0_mask), str(t_b0_mask)])
    subprocess.run(['rsync', '-L', str(transform_b0_to_t1), str(t_transform_b0_to_t1)])
    subprocess.run(['rsync', '-L', str(transform_t1_to_MNI_affine), str(t_transform_t1_to_MNI_affine)])
    subprocess.run(['rsync', '-L', str(transform_t1_to_MNI_warp), str(t_transform_t1_to_MNI_warp)])

    # skull-stripping
    t_fa_ss = t_folder / 'fa_skullstrip.nii.gz'
    if not t_fa_ss.is_file():
        subprocess.run(['fslmaths', str(t_fa), '-mul', str(t_b0_mask), str(t_fa_ss)])
    t_md_ss = t_folder / 'md_skullstrip.nii.gz'
    if not t_md_ss.is_file():
        subprocess.run(['fslmaths', str(t_md), '-mul', str(t_b0_mask), str(t_md_ss)])
    
    # apply transformations
    if transformation == 'affine':
        list_transforms = [t_transform_t1_to_MNI_affine, t_transform_b0_to_t1]
    elif transformation == 'nonrigid':
        list_transforms = [t_transform_t1_to_MNI_warp, t_transform_t1_to_MNI_affine, t_transform_b0_to_t1]
    else:
        raise ValueError(f'Unknown transformation: {transformation}. Should be either "affine" or "nonrigid".')
    t_fa_ss_mni = t_folder.parent / 'fa_skullstrip_MNI152.nii.gz'
    t_md_ss_mni = t_folder.parent / 'md_skullstrip_MNI152.nii.gz'
    apply_transform_to_img_in_b0(t_fa_ss, path_MNI152, t_fa_ss_mni, list_transforms)
    apply_transform_to_img_in_b0(t_md_ss, path_MNI152, t_md_ss_mni, list_transforms)

    # remove the tmp folder
    shutil.rmtree(t_folder)

def generate_job_tuples(spreadsheet_csv, output_dir, transformation='affine'):
    df = pd.read_csv(spreadsheet_csv)
    list_job_tuples = []

    for _, row in df.iterrows():
        job_tuple = (row['path_wmatlas'], output_dir, row['dataset'], row['subject'], row['session'], row['scan'], transformation)
        list_job_tuples.append(job_tuple)
    
    print(f'Generated {len(list_job_tuples)} job tuples.')
    return list_job_tuples

def crop_downsample_all(dir, input_suffix):
    # find nifti images to crop and downsample
    list_nifti = [fn for fn in Path(dir).glob(f'**/*{input_suffix}')]
    print(f'Found {len(list_nifti)} images to crop and downsample.')

    # crop and downsample
    dataset = CropDownSampleDataset(list_nifti, dir)
    dataloader = DataLoader(
        dataset=dataset, batch_size=4, shuffle=False, 
        num_workers=4, pin_memory = False, prefetch_factor=None)
    for _ in tqdm(dataloader, total=len(dataloader), desc=f'Crop and downsample NIFTIs in {dir}'):
        pass

if __name__ == '__main__':
    # user input
    transformation = 'affine'  # either 'affine' or 'nonrigid'
    ##

    prepare_braid_style_spreadsheet(output_csv='experiments/2024-06-06_VMAP_Model_Inference/vmap_braid_style.csv')
    
    list_job_tuples = generate_job_tuples(
        spreadsheet_csv = 'experiments/2024-06-06_VMAP_Model_Inference/vmap_braid_style.csv', 
        output_dir = f'/home-local/gaoc11/VMAP_Brain_Age/WMage_{transformation}', 
        transformation=transformation)
    with Pool(processes=6) as pool:
        list(tqdm(pool.imap(get_skull_stripped_fa_md_in_mni152, list_job_tuples, chunksize=1), total=len(list_job_tuples), desc=f'WMage_{transformation}'))
    
    crop_downsample_all(
        dir = f'/home-local/gaoc11/VMAP_Brain_Age/WMage_{transformation}', 
        input_suffix = '_skullstrip_MNI152.nii.gz',
        )