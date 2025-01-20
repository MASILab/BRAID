from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    CenterSpatialCrop,
    Spacing,
    SaveImage,
)

mni152 = 'data/template/MNI_152.nii.gz'

transform = Compose([
    LoadImage(reader="NibabelReader", image_only=True),
    EnsureChannelFirst(),
    Orientation(axcodes="RAS"),
    CenterSpatialCrop(roi_size=(192, 228, 192)),
    Spacing(pixdim=(1.5, 1.5, 1.5), mode='bilinear'), # expected: 128 x 152 x 128, 1.5mm^3
    SaveImage(
        output_postfix='crop_downsample', 
        output_ext='.nii.gz', 
        output_dir='experiments/2024-12-05_GradCam_affine_vs_nonrigid/data', 
        separate_folder=False,
        print_log=False,
        ),
])
transform(mni152)