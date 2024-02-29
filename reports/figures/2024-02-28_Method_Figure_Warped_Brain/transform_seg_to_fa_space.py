import subprocess
from pathlib import Path
from tqdm import tqdm

def transform_seg(move, ref, out, transforms):
    if not Path(out).parent.is_dir():
        subprocess.run(['mkdir', '-p', str(Path(out).parent)])
        
    command = [
        'antsApplyTransforms', '-d', '3',
        '-i', str(move),
        '-r', str(ref), 
        '-o', str(out), 
        '-n', 'nearestNeighbor', 
        ]
    for transform in transforms:
        command += ['-t', str(transform)]
        
    subprocess.run(command)


if __name__ == '__main__':
    images_dir = Path('reports/figures/2024-02-28_Method_Figure_Warped_Brain/images')

    list_fa_mni152 = sorted(images_dir.glob('**/fa_skullstrip_MNI152.nii.gz'))            
    for fa_mni152 in tqdm(list_fa_mni152):
        # input
        seg = sorted(fa_mni152.parent.glob("sub-*_T1w_seg.nii.gz"))[0]
        fa_mni152_warped = str(fa_mni152).replace('.nii.gz', '_warped.nii.gz')
        t_affine = fa_mni152.parent / 'transform_t1toMNI_affine.mat'
        t_warp = fa_mni152.parent / 'transform_t1toMNI_warp.nii.gz'
        
        # transformations
        transforms_mni152 = [t_affine]
        transforms_mni152_warped = [t_warp, t_affine]
        
        # output
        seg_mni152 = str(seg).replace('.nii.gz', '_MNI152.nii.gz')
        seg_mni152_warped = str(seg).replace('.nii.gz', '_MNI152_warped.nii.gz')

        # apply transformations
        transform_seg(move=seg, ref=fa_mni152, out=seg_mni152, transforms=transforms_mni152)
        transform_seg(move=seg, ref=fa_mni152_warped, out=seg_mni152_warped, transforms=transforms_mni152_warped)
