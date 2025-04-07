<p align="center">
    <img src="figures/logo.PNG" alt="braid" width="300">
</p>

# BRAID: Brain Age Identification from Diffusion MRI

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

BRAID estimates brain age from diffusion MRI. 
Unlike its counterparts, which typically use anatomical features (such as volume and shape of brain regions) to assist in the estimation, BRAID deliberately *destroys* anatomical information.
Through non-rigid transformations, BRAID minimizes anatomical information to better focus on subtle microstructural changes that predate apparent anatomical changes in neurodegeneration. 
The goal of BRAID is not to be a perfect chronological age estimator, but to provide an earlier biomarker for neurodegenerative disease prediction.

<div align="center">
    <img src="figures/warp_brain.png" alt="warp brain" width="650"/>
</div>

Illustration of the preprocessing, which minimizes anatomical information through non-rigid transformations.

<div align="center">
    <img src="figures/preprocessing.png" alt="preprocessing" width="650"/>
</div>

## Citations
If you use BRAID in your research, please cite the following papers:
- Gao et al. "Brain age identification from diffusion MRI synergistically predicts neurodegenerative disease." *Imaging Neuroscience* (2025) [[arXiv preprint](https://arxiv.org/abs/2410.22454)]
- Gao et al. "Predicting age from white matter diffusivity with residual learning." *Medical Imaging 2024: Image Processing.* (2024) [https://doi.org/10.1117/12.3006525](https://doi.org/10.1117/12.3006525)


## How to use BRAID
### Method 1 (recommended): Use the Singularity container

Provide the inputs and hit the "run button". It's that simple! There is no need to install dependencies by yourself. This is the recommended way for users who want to quickly test BRAID on their own datasets.

Please find the Singularity image and instructions on how to use it on the [Zenodo repository](https://zenodo.org/records/15091613).

### Method 2: Use the source code

For users who want to play around with the source code.

#### Step 1: Clone the repository

```bash
git clone https://github.com/MASILab/BRAID.git
```
#### Step 2: Create a virtual environment

Install python 
> Note: Development was done on Python 3.11.5. Tested on Python 3.12.8 as well. Please use either of the versions (or versions in between) for smoothest user experience.

```bash
cd BRAID
python -m venv env
source env/bin/activate
```

#### Step 3: Install dependencies

Install the (minimal) Python packages required for BRAID:
```bash
pip install .
```

Install other required software dependencies (for preprocessing):

| Package                                          | Tested version |
|--------------------------------------------------|----------------|
| [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL)| 6.0.4          |
| [MRtrix](https://www.mrtrix.org/)                | 3.0.3          |
| [c3d](https://sourceforge.net/projects/c3d/files/c3d/1.0.0/) | 1.0.0  |
| [ANTs](https://stnava.github.io/ANTs/)           | 2.3.1.dev48-g39ce5 |

#### Step 4: Download model weights from Hugging Face

> Link: [braid-v1.0 weights](https://huggingface.co/chenyugoal/braid-v1.0/tree/main)

```bash
# if git-lfs is not installed yet
git lfs install
# clone the repository to the current directory
git clone https://huggingface.co/chenyugoal/braid-v1.0
```

#### Example usage: Model inference with one line of command

```bash
braid_one_sample_inference -d {path to dwi} -v {path to bval} -c {path to bvec} -t {path to t1 image} -tm {path to t1 brain segmentation or brain mask} -m ./data/template/MNI_152.nii.gz -w {path to the braid-v1.0 directory} -i -o {path to the output directory}
```

See descriptions about the arguments by running `braid_one_sample_inference -h`.


## FAQ
- **Q: I saw "WM age nonlinear" in some scripts. What is it?**
- A: The name "WM age nonlinear" appears most likely in the legacy scripts created during model development. We had a debate whether we should use "WM age nonlinear" or "WM age nonrigid" when referring to the brain age estimation made based on the "warped" images. At the time point of writing the journal paper, we leaned more  towards "WM age nonrigid" and continued using it afterwards.
