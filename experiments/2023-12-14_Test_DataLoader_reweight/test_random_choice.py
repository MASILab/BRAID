"""
Simulation for testing random.choice()

Author: Chenyu Gao
Date: Dec 14, 2023
"""

import random
import matplotlib.pyplot as plt
from tqdm import tqdm

min = 45
max = 90

max_iter = 1000000
milestones = [100, 1000, 10000, 100000, 1000000]
fig, axes = plt.subplots(1, len(milestones), sharex=True, figsize=(5*len(milestones), 5))
idx = 0

ages = []
for i in tqdm(range(1000000)):
    ages.append(random.choice(range(min, max)))
    if (i+1) in milestones:
        axes[idx].hist(ages, bins=range(40, 101), edgecolor='black', color='blue')
        axes[idx].set_xlabel('Age')
        axes[idx].set_ylabel('Count')
        axes[idx].set_title(f'#iteration: {i+1}')
        idx += 1

fig.savefig(f'experiments/2023-12-14_Test_DataLoader_reweight/figs/test_random_choice.png', bbox_inches='tight')
plt.close('all')