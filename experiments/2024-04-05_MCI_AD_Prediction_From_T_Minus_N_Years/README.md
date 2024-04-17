# Code for the "T-0, T-1, ..., T-n" MCI/AD prediction experiment.

We have five different brain age estimation models. We predict the brain ages on scans from about six thousand sessions. Among the six thousand sessions, there are cognitively normal (CN), MCI, and AD samples. We would like to investigate how predictive each brain age (and its combination with each other) is of MCI/AD before the diagnosis. Moreover, we want to investigate how the prediction capability of each brain age changes as we make the prediction earlier and earlier. For example, make prediction one year before the MCI diagnosis v.s. make prediction five years before the MCI diagnosis.

The code in this folder implements the "index cutting approach" (which is how I call it) for sampling subsets, each representing a cohort that is n years before MCI/AD diagnosis. There are some issues that need to be fixed in the next version:

1. The "index cutting approach" provides a consistent number of samples across subsets and allows us to use almost all samples of qualified subjects. We can perform five-folds cross validation with that amount of data. However, given that subjects have different number and length of longitudinal visits, the distributions of the sampled subsets will have very different widths. In particular, "T-5" subset will have a wider distribution of data samples (some data points at 1 year before MCI while some at 10 years before MCI for instance) than "T-1" subset. As the result, the definition of each subset becomes unclear. We used the "average time to MCI/AD", but it might potentially mislead readers. This is a huge disadvantage in terms of interpretability.

2. The five-folds splitting does not take the pairing of MCI/AD and CN samples into consideration. After matching CN samples in terms of age and sex, the splitting shuffles all samples and sample each fold on the subject level rather than on the pair level. We should do the splitting at the pair level (using match_id).

3. Subjects who turned back to cognitively normal from MCI/AD are included. These data points should be excluded.
