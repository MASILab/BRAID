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

BRAID predicts brain age from diffusion MRI. 
Unlike its counterparts, which typically use anatomical features (volume, shape, etc.) to assist in the prediction, BRAID deliberately destroys anatomical information.
Through non-rigid transformations, BRAID minimizes anatomical information to better focus on subtle microstructural changes that predate apparent anatomical changes in neurodegeneration. 
The goal of BRAID is not to be a perfect chronological age predictor, but to provide an earlier biomarker for neurodegenerative disease prediction.

<div align="center">
    <img src="figures/warp_brain.png" alt="warp brain" width="650"/>
</div>

Illustration of the preprocessing, which minimizes anatomical information through non-rigid transformations.

<div align="center">
    <img src="figures/preprocessing.png" alt="preprocessing" width="650"/>
</div>

- Publications:
    - [Gao et al. "Brain age identification from diffusion MRI synergistically predicts neurodegenerative disease." *arXiv* (2024)](https://arxiv.org/abs/2410.22454)
    - [Gao et al. "Predicting age from white matter diffusivity with residual learning." *Medical Imaging 2024: Image Processing.* (2024)](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12926/129262I/Predicting-age-from-white-matter-diffusivity-with-residual-learning/10.1117/12.3006525.full)

- Citation:
    ```bibtex
    @misc{gao2024brainageidentificationdiffusion,
        title={Brain age identification from diffusion MRI synergistically predicts neurodegenerative disease}, 
        author={Chenyu Gao and Michael E. Kim and Karthik Ramadass and Praitayini Kanakaraj and Aravind R. Krishnan and Adam M. Saunders and Nancy R. Newlin and Ho Hin Lee and Qi Yang and Warren D. Taylor and Brian D. Boyd and Lori L. Beason-Held and Susan M. Resnick and Lisa L. Barnes and David A. Bennett and Katherine D. Van Schaik and Derek B. Archer and Timothy J. Hohman and Angela L. Jefferson and Ivana Išgum and Daniel Moyer and Yuankai Huo and Kurt G. Schilling and Lianrui Zuo and Shunxing Bao and Nazirah Mohd Khairi and Zhiyuan Li and Christos Davatzikos and Bennett A. Landman},
        year={2024},
        eprint={2410.22454},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2410.22454}, 
    }

    @inproceedings{10.1117/12.3006525,
        title = {{Predicting age from white matter diffusivity with residual learning}},
        author = {Chenyu Gao and Michael E. Kim and Ho Hin Lee and Qi Yang and Nazirah Mohd Khairi and Praitayini Kanakaraj and Nancy R. Newlin and Derek B. Archer and Angela L. Jefferson and Warren D. Taylor and Brian D. Boyd and Lori L. Beason-Held and Susan M. Resnick and Yuankai Huo and Katherine D. Van Schaik and Kurt G. Schilling and Daniel Moyer and Ivana Išgum and Bennett A. Landman},
        volume = {12926},
        booktitle = {Medical Imaging 2024: Image Processing},
        editor = {Olivier Colliot and Jhimli Mitra},
        organization = {International Society for Optics and Photonics},
        publisher = {SPIE},
        pages = {129262I},
        keywords = {brain age, diffusion tensor imaging, deep learning, convolutional neural networks},
        year = {2024},
        doi = {10.1117/12.3006525},
        URL = {https://doi.org/10.1117/12.3006525}
    }
    ```

## Using the code:
### 1. Installation and pretrained weights
#### 1.1 Clone the repository
```bash
git clone https://github.com/MASILab/BRAID.git
```
#### 1.2 Create a virtual environment (optional, recommended)

Install python 
> Note: Development was done on Python 3.11.5. Tested on Python 3.12.8 as well. Please use either of the versions (or versions in between) for smoothest user experience.

```bash
cd BRAID
python -m venv env
source env/bin/activate
```

#### 1.3 Install dependencies

```bash
pip install .
```

**Other dependencies (for preprocessing):**

| Package                                          | Tested version |
|--------------------------------------------------|----------------|
| [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL)| 6.0.4          |
| [MRtrix](https://www.mrtrix.org/)                | 3.0.3          |
| [c3d](https://sourceforge.net/projects/c3d/files/c3d/1.0.0/) | 1.0.0  |
| [ANTs](https://stnava.github.io/ANTs/)           | 2.3.1.dev48-g39ce5 |

#### 1.4 Download model weights from Hugging Face

> Link: [braid-v1.0 weights](https://huggingface.co/chenyugoal/braid-v1.0/tree/main)

```bash
# if git-lfs is not installed yet
git lfs install
# clone the repository to the current directory
git clone https://huggingface.co/chenyugoal/braid-v1.0
```

### 2. Model inference with one line of command

```bash
braid_one_sample_inference -d {path to dwi} -v {path to bval} -c {path to bvec} -t {path to t1 image} -tm {path to t1 brain segmentation or brain mask} -m ./data/template/MNI_152.nii.gz -w {path to the braid-v1.0 directory} -i -o {path to the output directory}
```

See descriptions about the arguments by running `braid_one_sample_inference -h`.

## FAQ
- **Q: I saw "WM age nonlinear" in some scripts. What is it?**
- A: The name "WM age nonlinear" appears most likely in the legacy scripts created during model development. We had a debate whether we should use "WM age nonlinear" or "WM age nonrigid" when referring to the brain age estimation made based on the "warped" images. At the time point of writing the journal paper, we leaned more  towards "WM age nonrigid" and continued using it afterwards.
