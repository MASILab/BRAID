## Probability Decay

Models trained before 2023-12-14, i.e., `models/2023-12-11_ResNet18` and `models/2023-12-12_ResNet50`, were trained using the dataloader without the probability decay feature, which means that in each iteration, images that have been sampled in previous iterations would still have the same probability of being sampled as those that have not been sampled in previous iterations. The consequence of this equal-probability random sampling is that it takes a big number of iterations to get all images used at least once. See `experiments/2023-12-13_Test_DataLoader` for the empirical experiments about the phenomenon.

Models afterwards were trained using the updated dataloader (or updated upstream functions) with the probability decay functionality such that it takes less iterations to cover all samples in the training set. See `experiments/2023-12-14_Test_DataLoader_reweight` for the empirical experiments on the dataloader with probability decay.

## Types of Models

### FA+MD, affine, skull-strip

- **models/2023-12-11_ResNet18**
- **models/2023-12-12_ResNet50**
- **models/2023-12-14_ResNet18**
- **models/2023-12-21_ResNet50**
- **models/2023-12-22_ResNet101**
- **models/2023-12-27_ResNet152**
- **models/2024-01-16_ResNet101_MLP**

### FA+MD, affine+SyN, skull-strip

- **models/2024-02-07_ResNet101_BRAID_warp**

### T1w, affine, skull-strip

- **models/2024-02-06_T1wAge_ResNet101**: trained with data after the PNG QA (500 T1w randomly sampled from each dataset, totaling 6000) summarized in `data/databank_t1w/quality_assurance/2024-02-05_brain_affine`. There are T1w images of bad quality remaining in the training set.
- **models/2024-02-07_T1wAge_ResNet101**: trained with data after the QA summarized in `data/databank_t1w/quality_assurance/2024-02-05_brain_affine`, and the QA summarized in `data/databank_t1w/quality_assurance/2024-02-07_brain_affine_NACC_OASIS4_all`.

### T1w, affine, non-skull-strip
