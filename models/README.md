Models before 2023-12-14 were trained with the dataloader without the probability decay feature.

Starting from 2023-12-14_ResNet18, the dataloader (or its upstream functions) is written with the probability decay functionality such that it takes less batches to cover all samples in the training set.

Timeline for model training:
- 2023-12-14_ResNet18 Fold-1 **done**
- 2023-12-21_ResNet50 Fold-1 **done**
- 2023-12-22_ResNet101 Fold-1 **done**
- 2023-12-27_ResNet152 Fold-1 **done**
- 2023-12-22_ResNet101 Fold-2/3/4/5 **done**
- 2023-12-27_ResNet152 Fold-2/3/4/5 **done**
- 2023-12-21_ResNet50 Fold-2/3/4/5 **ETA: Jan 15**
- 