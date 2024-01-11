"""
Script for model inference.

Author: Chenyu Gao
Date: Jan 11, 2024
"""

import yaml
import argparse
import torch
import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from braid.evaluations import load_trained_model
from braid.dataset import get_the_sequence_of_scans, get_BRAID_dataloader
torch.set_flush_denormal(True)
device = torch.device('cuda')

if __name__ == "__main__":  

    # User inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, help="Path to yaml file, example: ./test_config.yaml")
    with open(parser.parse_args().yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    # Test dataset/loader
    df = pd.read_csv(config['dataset']['csv_file'])
    
    list_scans_test = get_the_sequence_of_scans(
        csv_file = config['dataset']['csv_file'],
        subjects = 'all',
        age_min = config['dataset']['age_min'],
        age_max = config['dataset']['age_max'],
        mode = 'test'
    )
    
    dataloader_test = get_BRAID_dataloader(
        dataset_root = config['dataset']['root'],
        csv_file = config['dataset']['csv_file'],
        list_scans = list_scans_test,
        batch_size = config['batch_size']
    )
    
    # Test model from each fold
    for fold_idx in config['model']['fold']:
        print(f'Fold-{fold_idx}:')
        # load model
        weights_folder = Path(config['model']['weights_root']) / f'fold-{fold_idx}'
        path_pth = [fn for fn in weights_folder.iterdir() if fn.suffix == '.pth'][0]
        
        model  = load_trained_model(
            model_name = config['model']['name'],
            mlp_hidden_layer_sizes = config['model']['mlp_hidden_layer_sizes'], 
            feature_vector_length = config['model']['feature_vector_length'], 
            path_pth = path_pth,
        )
        
        # record gt and predicted age
        print('Model inference...')
        list_gt_all = []
        list_pred_all = []
        
        model.eval()
        with torch.no_grad():
            for images, label_feature, age in tqdm(dataloader_test):
                images, label_feature = images.to(device, non_blocking=True), label_feature.to(device, non_blocking=True)
                
                gt = age
                pred = model(images, label_feature).detach().cpu()
                
                list_gt_all += torch.flatten(gt).tolist()
                list_pred_all += torch.flatten(pred).tolist()
        
        # Save predictions to csv
        df.insert(0, "age_gt", list_gt_all)
        df.insert(1, "age_pred", list_pred_all)
        
        save_csv = Path(config['output_dir']) / f'predicted_age_fold-{fold_idx}.csv'
        subprocess.run(['mkdir', '-p', str(save_csv.parent)])
        df.to_csv(save_csv, index=False)
