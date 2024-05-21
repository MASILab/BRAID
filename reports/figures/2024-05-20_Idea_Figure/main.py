import numpy as np
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from functions import data


def visualize_axial_slices(path_nifti, output_prefix, crop=10, skip_head=0, skip_tail=0, every=1, dpi=300):
    img = nib.load(path_nifti)
    img = nib.as_closest_canonical(img)
    data = np.squeeze(img.get_fdata())
    resolution = data.shape[:3]
    aspect = img.header.get_zooms()[1] / img.header.get_zooms()[0]

    if 'fa' in path_nifti.name:
        vmin, vmax = 0, 1
    elif 'md' in path_nifti.name:
        vmin, vmax = 0, 0.003
    else:
        vmin, vmax = None, None
    
    num_slices = (resolution[2] - skip_head - skip_tail) // every
    for i in tqdm(range(num_slices), desc='screenshots'):
        slice_index = skip_head + i * every
        fig, ax = plt.subplots()
        ax.imshow(
            data[crop:resolution[0]-crop, crop:resolution[1]-crop, slice_index].T,
            cmap='gray', origin='lower', aspect=aspect,
            interpolation='nearest', vmin=vmin, vmax=vmax,
        )
        ax.axis('off')
        
        save_png = f"{output_prefix}_{slice_index:03d}.png"
        Path(save_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_png, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close('all')


def visualize_colorbar(png):
    fig, ax = plt.subplots(figsize=(12, 2))
    fig.subplots_adjust(bottom=0.5)

    cmap = mpl.cm.gray
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    ax.axis('off')
    fig.savefig(png, dpi=300, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    dir='reports/figures/2024-05-20_Idea_Figure/data/'
    data.load_sample_and_symlink_data(dir)

    for path_nifti in Path(dir).glob('*/*/*.nii.gz'):
        output_prefix = str(path_nifti).replace('/data/', '/figs/').replace('.nii.gz', '')
        print(output_prefix)
        visualize_axial_slices(path_nifti, output_prefix)
    visualize_axial_slices(path_nifti=Path('/nfs2/ForChenyu/MNI_152.nii.gz'), output_prefix='reports/figures/2024-05-20_Idea_Figure/figs/MNI_152/MNI152')
    visualize_colorbar('reports/figures/2024-05-20_Idea_Figure/figs/colorbar.png')