import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

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
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24, 15))
    
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
    
    # Adjust the spacing    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)    
    fig.savefig(path_png, bbox_inches='tight')
    plt.close('all')


# TODO: test
# generate_qa_screenshot_fa_md(path_fa, path_md, path_png, offset=0)