import re
import os
import ants
import antspynet
import subprocess
import pandas as pd
from pathlib import Path

demog = pd.read_csv('data/subject_info/raw/VMAP_2024-05_update/Landman_Diffusion_Harmonization_23-24_data_withdiagnosis.csv', index_col=0)
        
dict_datasets = {
    'VMAP_JEFFERSON': 'JEFFERSON',
    'VMAP_2.0': 'JEFFERSONVMAP',
    'VMAP_TAP': 'JEFFERSONTAP'
}

# Continue from existing results / start from scratch
output_csv = 'experiments/2024-06-06_VMAP_Model_Inference/results/DeepBrainNet_predictions/vmap_brain_age_deepbrainnet_2024-06-06.csv'
if Path(output_csv).exists():
    results = pd.read_csv(output_csv)
else:
    dict_results = {
        'bids_subject_id': [],
        'bids_session_id': [],
        'bids_acquisition_label': [],
        'bids_run_label': [],
        'xnat_project': [],
        'vmac_id': [],
        'session_id': [],
        'age_chronological': [],
        'age_pred_dbn': [],
    }
    results = pd.DataFrame(dict_results)

# Model inference
checkpoint_count = 0
local_tmp_dir = Path('/tmp/.GoneAfterReboot')

for bids_dataset, xnat_project in dict_datasets.items():
    path_dataset = Path('/nfs2/harmonization/BIDS') / bids_dataset
    subject_folders = [f for f in path_dataset.iterdir() if f.is_dir() and f.name.startswith('sub-')]

    for subject in subject_folders:
        bids_subject_id = subject.name
        vmac_id = int(bids_subject_id.split('-')[1])
        session_folders = [f for f in subject.iterdir() if f.is_dir() and f.name.startswith('ses-')]

        for session in session_folders:
            bids_session_id = session.name
            session_id = bids_session_id.split('-')[1].split('x')[0]

            match = demog.loc[(demog['vmac_id'] == vmac_id)&(demog['session_id']==session_id), ]
            if len(match.index) == 1:
                age_chronological = match['age'].values[0]
            elif len(match.index) == 0:
                try:
                    age_chronological = demog.loc[(demog['vmac_id'] == vmac_id)&(demog['session_id']==session_id[:6]), 'age'].values[0]
                except:
                    print(f'No match for {bids_subject_id} {bids_session_id}')
                    age_chronological = None
            else:
                raise ValueError(f'Multiple matches for {bids_subject_id} {bids_session_id}')

            for t1w in session.glob('anat/*_T1w.nii.gz'):
                pattern = r'(sub-\w+)(?:_(ses-\w+))?(?:_(acq-\w+))?(?:_(run-\d{1,2}))?_T1w'
                matches = re.findall(pattern, t1w.name)
                _, _, bids_acquisition_label, bids_run_label = matches[0]
                
                # Check if this t1w has been processed
                exist_match = results.loc[
                    (results['bids_subject_id'] == bids_subject_id)&
                    (results['bids_session_id'] == bids_session_id)&
                    (results['bids_acquisition_label'] == bids_acquisition_label)&
                    (results['bids_run_label'] == bids_run_label), ]
                if len(exist_match.index) > 0:
                    if pd.notnull(exist_match['age_pred_dbn'].values[0]):
                        continue
                
                # prepare a local copy of the t1w
                t1w_local = local_tmp_dir / t1w.name
                try:
                    subprocess.run(['rsync', '-L', str(t1w), str(t1w_local)])
                except:
                    print(f'data transfer failed: {t1w}')
                
                if not t1w_local.is_file():
                    print(f"File not found: {t1w_local}")
                    continue

                image = ants.image_read(str(t1w_local))
                deep = antspynet.utilities.brain_age(
                    image,
                    do_preprocessing=True,
                    antsxnet_cache_directory='/tmp/.GoneAfterReboot/antsxnet_cache',
                    )
                
                result_row = {
                    'bids_subject_id': [bids_subject_id],
                    'bids_session_id': [bids_session_id],
                    'bids_acquisition_label': [bids_acquisition_label],
                    'bids_run_label': [bids_run_label],
                    'xnat_project': [xnat_project],
                    'vmac_id': [vmac_id],
                    'session_id': [session_id],
                    'age_chronological': [age_chronological],
                    'age_pred_dbn': [deep['predicted_age']],
                }
                result_row = pd.DataFrame(result_row)
                results = pd.concat([results, result_row], ignore_index=True)
                os.remove(t1w_local)  # remove local file
                print(f'{xnat_project} {vmac_id} {session_id} {age_chronological} {deep["predicted_age"]}')

                # Save checkpoint
                checkpoint_count += 1
                if checkpoint_count % 25 == 0:
                    results.to_csv(output_csv, index=False)
results.to_csv(output_csv, index=False)
