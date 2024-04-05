""" DeepBrainNet

Full citation: Vishnu M Bashyam, Guray Erus, Jimit Doshi, Mohamad Habes, Ilya M Nasrallah, Monica Truelove-Hill, 
Dhivya Srinivasan, Liz Mamourian, Raymond Pomponio, Yong Fan, Lenore J Launer, Colin L Masters, Paul Maruff, 
Chuanjun Zhuo, Henry Völzke, Sterling C Johnson, Jurgen Fripp, Nikolaos Koutsouleris, Theodore D Satterthwaite, 
Daniel Wolf, Raquel E Gur, Ruben C Gur, John Morris, Marilyn S Albert, Hans J Grabe, Susan Resnick, R Nick Bryan, 
David A Wolk, Haochang Shou, Christos Davatzikos, on behalf of the ISTAGING Consortium, the Preclinical Alzheimer's 
disease Consortium, ADNI, and CARDIA studies, 
MRI signatures of brain age and disease over the lifespan based on a deep brain network and 14468 individuals worldwide, 
Brain, Volume 143, Issue 7, July 2020, Pages 2312-2324, https://doi.org/10.1093/brain/awaa160

Original GitHub: https://github.com/vishnubashyam/DeepBrainNet

Implementation: https://github.com/ANTsX/ANTsPyNet

Citation for the implementation: Nicholas J. Tustison, Philip A. Cook, Andrew J. Holbrook, Hans J. Johnson, 
John Muschelli, Gabriel A. Devenyi, Jeffrey T. Duda, Sandhitsu R. Das, Nicholas C. Cullen, Daniel L. Gillen, 
Michael A. Yassa, James R. Stone, James C. Gee, and Brian B. Avants for the Alzheimer's Disease Neuroimaging Initiative. 
The ANTsX ecosystem for quantitative biological and medical imaging. Scientific Reports. 11(1):9068, Apr 2021.
"""

import os
import ants
import antspynet
import subprocess
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# User input:
dict_csv = {
    'input': {
        'test': '/tmp/.GoneAfterReboot/spreadsheet/t1wagepredict_test.csv',
        'train': '/tmp/.GoneAfterReboot/spreadsheet/t1wagepredict_train.csv',
    },
    'output': {
        'test': '/nfs/masi/gaoc11/projects/BRAID/models/2024-04-04_DeepBrainNet/predictions/predicted_age_test.csv',
        'train': '/nfs/masi/gaoc11/projects/BRAID/models/2024-04-04_DeepBrainNet/predictions/predicted_age_train.csv',
    },
}
databank_root_server = Path('/home/gaoc11/GDPR/masi/gaoc11/BRAID/data/databank_t1w/')  # path on hickory (GDPR sshfs)
local_tmp_dir = Path('/tmp/.GoneAfterReboot/tmp_t1')
local_tmp_dir.mkdir(parents=True, exist_ok=True)

# Model inference
for stage, csv in dict_csv['input'].items():
    df = pd.read_csv(csv)
    df['age_pred'] = None
    
    for i, row in tqdm(df.iterrows(), total=len(df.index), desc=f'Predict age for {stage} set'):
        path_t1_server = databank_root_server / row['dataset'] / row['subject'] / row['session'] / f"scan-{row['scan']}" / f"{row['dataset']}_{row['subject']}_{row['session']}_scan-{row['scan']}_T1w.nii.gz"

        # transfer to local
        path_t1 = local_tmp_dir / path_t1_server.name
        try:
            subprocess.run(['rsync', '-L', f'hickory:{str(path_t1_server)}', str(path_t1)])
        except:
            print(f'data transfer failed: {path_t1_server}')

        if not path_t1.is_file():
            print(f"File not found: {path_t1}")
            continue

        image = ants.image_read(str(path_t1))
        deep = antspynet.utilities.brain_age(
            image,
            do_preprocessing=True,
            antsxnet_cache_directory='/tmp/.GoneAfterReboot/antsxnet_cache',
            )
        df.at[i, 'age_pred'] = deep['predicted_age']
        df.to_csv(dict_csv['output'][stage], index=False)

        # remove local file
        os.remove(path_t1)