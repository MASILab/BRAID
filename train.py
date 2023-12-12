"""
Train the model using the configuration in yaml file.

Author: Chenyu Gao
Date: Dec 11, 2023
"""

import gc
import yaml
import argparse
import torch
import subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from braid.dataset import BRAID_Dataset
from braid.models import get_the_resnet_model
from braid.utls import generate_png_during_training
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":  
    device = torch.device('cuda')
    
    # User input: yaml file of the training configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, help="Path to yaml file, example: ./models/2023-12-11_ResNet50/train_config.yaml")
    with open(parser.parse_args().yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    for fold_idx in config['model']['fold']:
        
        # subjects splitting (train/val)
        crossval_subjects_dir = Path(config['dataset']['cv_subjects_dir'])
        try:
            subjects_train = np.load((crossval_subjects_dir / f"subjects_fold_{fold_idx}_train.npy"), allow_pickle=True)
            subjects_val = np.load((crossval_subjects_dir / f"subjects_fold_{fold_idx}_val.npy"), allow_pickle=True)
        except FileNotFoundError:
            print(f"Fold-{fold_idx} does not have valid subjects splitting.")
            continue
        
        # Dataloader
        dataset_train = BRAID_Dataset(
            dataset_root = config['dataset']['root'],
            csv_file = config['dataset']['csv_file'],
            subjects = subjects_train,
            age_min = config['dataset']['age_min'],
            age_max = config['dataset']['age_max'],
            mode = 'train',
        )
        dataset_val = BRAID_Dataset(
            dataset_root = config['dataset']['root'],
            csv_file = config['dataset']['csv_file'],
            subjects = subjects_val,
            age_min = config['dataset']['age_min'],
            age_max = config['dataset']['age_max'],
            mode = 'test',
        )
        dataloader_train = DataLoader(
            dataset_train, 
            batch_size = config['optimization']['batch_size'], 
            shuffle = True, 
            num_workers = config['optimization']['num_workers'],
            pin_memory = True
            )
        dataloader_val = DataLoader(
            dataset_val, 
            batch_size = config['optimization']['batch_size'], 
            shuffle = False, 
            num_workers = config['optimization']['num_workers'],
            pin_memory = True
            )

        # Model
        model = get_the_resnet_model(
            model_name = config['model']['name'], 
            feature_vector_length = config['model']['feature_vector_length'], 
            MLP_hidden_layer_sizes = config['model']['mlp_hidden_layer_sizes']
            ).to(device, non_blocking=True)
        
        # Optimizer
        if config['optimization']['optimizer']['name'] == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr = config['optimization']['optimizer']['lr'],
            )
        else:
            raise ValueError("Optimizer not supported yet.")
        
        # Learning rate scheduler
        if config['optimization']['lr_scheduler']['name'] == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer = optimizer,
                max_lr = config['optimization']['lr_scheduler']['lr_max'],
                epochs = config['optimization']['epochs'],
                steps_per_epoch = round(len(dataset_train)/(config['optimization']['batch_size']*config['optimization']['batch_size_mimic_factor'])),
                cycle_momentum = config['optimization']['lr_scheduler']['cycle_momentum'],
            )
        else:
            raise ValueError("LR scheduler not supported yet.")
        
        # Loss function
        if config['optimization']['loss'] == 'MSELoss':
            loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Loss function not supported yet.")
        
        # Tensorboard and model weights saving
        writer = SummaryWriter(log_dir = (Path(config['output']['tensorboard']) / f"{config['model']['name']}_fold-{fold_idx}"))
        best_val_loss = float('inf')
        weights_local_dir = Path(config['output']['weights_local_dir']) / f"fold-{fold_idx}"
        weights_server_dir = Path(config['output']['weights_server_dir']) / f"fold-{fold_idx}"
        subprocess.run(['mkdir', '-p', weights_local_dir])
        subprocess.run(['mkdir', '-p', weights_server_dir])

        print(f"Training fold-{fold_idx} of model {config['model']['name']}...")
        for epoch in range(config['optimization']['epochs']):
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()
            for i, (images, label_feature, age) in enumerate(tqdm(dataloader_train)):
                images, label_feature, age = images.to(device, non_blocking=True), label_feature.to(device, non_blocking=True), age.to(device, non_blocking=True)
                
                if (i+1) % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    if config['output']['png_sanity_check'] != '':
                        generate_png_during_training(
                            img_tensor = images[0,:,:,:,:].cpu(), 
                            path_png = Path(config['output']['png_sanity_check']) / f"fold-{fold_idx}_epoch-{epoch}_batch-{i}_age-{age[0].cpu().numpy():.1f}.png"
                        )
                            
                with torch.autocast('cuda'):
                    output = model(images, label_feature)
                    loss = loss_fn(output, age.view(-1, 1))
                    epoch_loss += loss.detach()

                loss /= config['optimization']['batch_size_mimic_factor']
                loss.backward()

                if (i+1) % config['optimization']['batch_size_mimic_factor'] == 0 or (i+1) == len(dataloader_train):
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    
            epoch_loss = epoch_loss / len(dataloader_train)
            writer.add_scalar('Train/Loss', epoch_loss, epoch)

            # validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, label_feature, age in tqdm(dataloader_val):
                    images, label_feature, age = images.to(device, non_blocking=True), label_feature.to(device, non_blocking=True), age.to(device, non_blocking=True)

                    output = model(images, label_feature)
                    loss_val = loss_fn(output, age.view(-1, 1))
                    
                    val_loss += loss_val.detach()

            val_loss = val_loss / len(dataloader_val)
            writer.add_scalar('Val/Loss', val_loss, epoch)

            print("Epoch: {}\tCurrent LR: {}\tTrain Loss: {}\tValidation Loss: {}".format(epoch, scheduler.get_last_lr(), epoch_loss, val_loss))
            
            # Save improved model to local
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                path_pth_local = weights_local_dir / f"{config['model']['name']}_fold-{fold_idx}_epoch-{epoch}_valloss-{best_val_loss:.4f}.pth"
                torch.save(model.state_dict(), path_pth_local)
                print(f'Saved improved model to {path_pth_local} at epoch {epoch} with validation loss {best_val_loss}\n')
        
        path_pth_server = weights_server_dir / f"{config['model']['name']}_fold-{fold_idx}_epoch-{epoch}_valloss-{best_val_loss:.4f}.pth"
        print(f"Copying best model of fold-{fold_idx} to {path_pth_server}\n")
        subprocess.run(['cp', path_pth_local, path_pth_server])
        
        writer.close()





