# Author: Michael Kim, Chenyu Gao
# Date: Nov 28, 2023

import subprocess
from pathlib import Path

def register_b0_to_MNI152(path_b0, path_t1, path_t1_brain, path_MNI152, outdir):
    """ 
    First, use epi_reg to register b0 to T1w. Then, affine register T1w to standard template MNI152.
    """

    temp_folder = Path(outdir) / 'temp'  # store tempory intermidiate files
    subprocess.run(['mkdir', '-p', str(temp_folder)])
    
    # transformation from b0 to T1w in fsl format
    b0_to_t1_fsl = temp_folder / 'b0_to_t1_fsl.mat'
    b0_to_t1_fsl_wo_ext = str(b0_to_t1_fsl).replace('.mat', '')

    t1_to_b0_fsl = temp_folder / 't1_to_b0_fsl.mat'
    
    command = ['epi_reg', '--epi={}'.format(path_b0), '--t1={}'.format(path_t1), '--t1brain={}'.format(path_t1_brain), '--out={}'.format(b0_to_t1_fsl_wo_ext)]
    subprocess.run(command)
    command = ['convert_xfm', '-omat', str(t1_to_b0_fsl), '-inverse', str(b0_to_t1_fsl)]
    subprocess.run(command)

    # transformation from b0 to T1w in ANTs format
    b0_to_t1_ants = temp_folder / 'b0_to_t1_ants.txt'
    t1_to_b0_ants = temp_folder / 't1_to_b0_ants.txt'

    command = ['c3d_affine_tool', '-ref', str(path_t1), '-src', str(path_b0), str(b0_to_t1_fsl), '-fsl2ras', '-oitk', str(b0_to_t1_ants)]
    subprocess.run(command)
    command = ['c3d_affine_tool', '-ref', str(path_b0), '-src', str(path_t1), str(t1_to_b0_fsl), '-fsl2ras', '-oitk', str(t1_to_b0_ants)]
    subprocess.run(command)

    # transformation from T1w to standard template (rigid + affine)
    t1_to_template_affine = temp_folder / 't1_to_MNI1520GenericAffine.mat'
    t1_to_template_prefix = str(t1_to_template_affine).replace('0GenericAffine.mat', '')
    command = ['antsRegistrationSyN.sh', '-d', '3', '-f', str(path_MNI152), '-m', str(path_t1), '-o', t1_to_template_prefix, '-t', 'a']
    subprocess.run(command)

    # copy selected files to output folder
    b0_to_t1_ants_copy_to = Path(outdir) / 'transform_b0tot1.txt'
    t1_to_b0_ants_copy_to = Path(outdir) / 'transform_t1tob0.txt'
    t1_to_template_affine_copy_to = Path(outdir) / 'transform_t1toMNI_affine.mat'

    subprocess.run(['cp', str(b0_to_t1_ants), str(b0_to_t1_ants_copy_to)])
    subprocess.run(['cp', str(t1_to_b0_ants), str(t1_to_b0_ants_copy_to)])
    subprocess.run(['cp', str(t1_to_template_affine), str(t1_to_template_affine_copy_to)])

    print('Finished all registrations. Cleaning up temp folder: {}'.format(temp_folder))
    subprocess.run(['rm', '-r', str(temp_folder)])


def apply_transform_to_img_in_b0(path_img_b0, path_img_ref, path_img_out, path_transform_b0_to_t1, path_transform_t1_to_MNI152):
    """
    Apply transformations to image in b0 space. This will align the image to the MNI152 standard template.
    path_img_b0: path to the image in b0 space
    path_img_ref: path to the reference image (MNI152 template image)
    path_img_out: path to the output image
    path_transform_b0_to_t1: path to the transformation from b0 to T1w (example: transform_b0tot1.txt)
    path_transform_t1_to_MNI152: path to the transformation from T1w to MNI152 (example: transform_t1toMNI_affine.mat)
    """
    if not Path(path_img_out).parent.is_dir():
        subprocess.run(['mkdir', '-p', str(Path(path_img_out).parent)])

    command = ['antsApplyTransforms', '-d', '3', 
               '-i', str(path_img_b0), 
               '-r', str(path_img_ref), 
               '-o', str(path_img_out), 
               '-n', 'Linear', 
               '-t', str(path_transform_t1_to_MNI152), '-t', str(path_transform_b0_to_t1)]
    subprocess.run(command)

# path_b0 = '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_dti/HCPA/sub-HCA6295782/ses-1/scan-1/firstshell/dwmri_firstshell_b0.nii.gz'
# path_t1 = '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_dti/HCPA/sub-HCA6295782/ses-1/scan-1/t1w/t1w.nii.gz'
# path_t1_brain = '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_dti/HCPA/sub-HCA6295782/ses-1/scan-1/t1w_brain_mask/t1w_brain.nii.gz'
# path_MNI152 = '/nfs2/ForChenyu/MNI_152.nii.gz'
# outdir = '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_dti/HCPA/sub-HCA6295782/ses-1/scan-1/transform'

# register_b0_to_MNI152(path_b0, path_t1, path_t1_brain, path_MNI152, outdir)

# path_final_test = '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_dti/HCPA/sub-HCA6295782/ses-1/scan-1/final/b0_MNI152.nii.gz'
# path_transform_b0_to_t1 = '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_dti/HCPA/sub-HCA6295782/ses-1/scan-1/transform/transform_b0tot1.txt'
# path_transform_t1_to_MNI152 = '/nfs/masi/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_dti/HCPA/sub-HCA6295782/ses-1/scan-1/transform/transform_t1toMNI_affine.mat'
# apply_transform_to_img_in_b0(path_b0, path_MNI152, path_final_test, path_transform_b0_to_t1, path_transform_t1_to_MNI152)