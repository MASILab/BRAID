### What we've learn from each round of classification experiments

#### v1
- CN vs. AD
  - Within each type of classifiers, the model using GM feature(s) always achieves the best performance.
  - The performance of naive Bayes does not rely heavily on the feature selection as other algorithms do.
  - RBF SVM's performance deteriorates when the data is high-dimensional. (ACC~=0.5)
- CN vs. MCI
  - Within each type of classifiers, the model using WM feature(s) or WM features (contaminated) achieves the best performance in most cases.
  - In version 2, we should automate the feature selection process and only report the performance on the best combination of features for each classifier type and each feature category (WM, GM, both, etc).