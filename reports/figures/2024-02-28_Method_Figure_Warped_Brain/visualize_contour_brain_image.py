import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
from skimage import measure

class brain_and_contour():
    def __init__(self, brain_nifti, seg_nifti, lut, output_dir):
        self.brain_nifti = brain_nifti
        self.seg_nifti = seg_nifti
        self.lut = lut
        self.output_dir = output_dir
        
    def load_image(self, path_nifti):
        img = nib.load(path_nifti)
        data = np.squeeze(img.get_fdata())
        return img, data

    def load_lut(self, path_lut):
        lut_data = np.genfromtxt(path_lut, skip_header=1, dtype=str)
        
        lut_indices = lut_data[:, 0].astype(int)
        lut_colors = lut_data[:, 2:].astype(int) / 255.0
        lut_colors[:, 3] = 1 - lut_colors[:, 3]
        lut_labels = lut_data[:, 1]
        return lut_indices, lut_colors, lut_labels
        
    def visualize(self,vmax,vmin,crop,skip_head,skip_tail,every,dpi,suffix):
        brain_img, brain_data = self.load_image(self.brain_nifti)
        seg_img, seg_data = self.load_image(self.seg_nifti)
        resolution = brain_data.shape[:3]
        assert nib.aff2axcodes(brain_img.affine)[2] == 'S', "Only support RAS orientation"
        aspect = brain_img.header.get_zooms()[1] / brain_img.header.get_zooms()[0]
        
        lut_indices, lut_colors, lut_labels = self.load_lut(self.lut)
        
        num_slices = (resolution[2] - skip_head - skip_tail) // every
        for i in tqdm(range(num_slices), desc='brain and contour'):
            slice_index = skip_head + i * every
            
            fig, ax = plt.subplots()
            ax.imshow(
                brain_data[crop:resolution[0]-crop, crop:resolution[1]-crop, slice_index].T,
                cmap='gray',
                origin='lower',
                aspect=aspect,
                interpolation='nearest',
                vmin=vmin,
                vmax=vmax,
            )
            
            for j, roi_index in enumerate(lut_indices):
                if roi_index in [0, 208, 209]:
                    continue
                roi_mask = np.where(seg_data[:, :, slice_index] == roi_index, 1, 0)
                contours = measure.find_contours(roi_mask, 0.5)

                for contour in contours:
                    ax.plot(contour[:, 0]-crop, contour[:, 1]-crop, linewidth=1.5, color=lut_colors[j, :])
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            save_png = Path(self.output_dir) / f'{slice_index:03d}_{suffix}.png'
            save_png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_png, dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close('all')


if __name__ == '__main__':
    images_dir = Path('reports/figures/2024-02-28_Method_Figure_Warped_Brain/images')
    list_fa_mni152 = sorted(images_dir.glob('**/fa_skullstrip_MNI152.nii.gz'))       
         
    for fa_mni152 in tqdm(list_fa_mni152):
        fa_mni152_warped = str(fa_mni152).replace('.nii.gz', '_warped.nii.gz')
        
        seg_mni152 = sorted(fa_mni152.parent.glob("sub-*_T1w_seg_MNI152.nii.gz"))[0]
        seg_mni152_warped = str(seg_mni152).replace('.nii.gz', '_warped.nii.gz')
        
        output_dir = str(fa_mni152.parent).replace('/images/', '/figs/')        
            
        # affine only
        b = brain_and_contour(
            brain_nifti=fa_mni152,
            seg_nifti=seg_mni152,
            lut='reports/figures/2024-02-28_Method_Figure_Warped_Brain/T1_TICV_seg.txt',
            output_dir=output_dir
        )
        b.visualize(vmax=1,vmin=0,crop=10,skip_head=60,skip_tail=80,every=2,dpi=300,suffix='MNI152_affine')
        
        # affine + warp
        b = brain_and_contour(
            brain_nifti=fa_mni152_warped,
            seg_nifti=seg_mni152_warped,
            lut='reports/figures/2024-02-28_Method_Figure_Warped_Brain/T1_TICV_seg.txt',
            output_dir=output_dir
        )
        b.visualize(vmax=1,vmin=0,crop=10,skip_head=60,skip_tail=80,every=2,dpi=300,suffix='MNI152_affine_warp')