"""
Train the model using the configuration in yaml file.

Author: Chenyu Gao
Date: Dec 11, 2023
"""

import gc
import yaml
import math
import argparse
import torch
import subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm
from braid.dataset import get_the_sequence_of_scans, get_dataloader
from braid.models import get_the_resnet_model
from braid.utls import generate_png_during_training
from torch.utils.tensorboard import SummaryWriter
torch.set_flush_denormal(True)
device = torch.device('cuda')

PNG_GENERATED = False

if __name__ == "__main__":  

    # User input: yaml file of the training configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, help="Path to yaml file, example: ./models/2023-12-11_ResNet50/train_config.yaml")
    with open(parser.parse_args().yaml, 'r') as f:
        config = yaml.safe_load(f)

    # Train the model for each fold
    for fold_idx in config['model']['fold']:
        
        # subjects splitting (train/val)
        crossval_subjects_dir = Path(config['dataset']['cv_subjects_dir'])
        try:
            subjects_train = np.load((crossval_subjects_dir / f"subjects_fold_{fold_idx}_train.npy"), allow_pickle=True)
            subjects_val = np.load((crossval_subjects_dir / f"subjects_fold_{fold_idx}_val.npy"), allow_pickle=True)
        except FileNotFoundError:
            print(f"Fold-{fold_idx} does not have valid subjects splitting.")
            continue
        
        # generate the sequence of samples (uniformly-distributed for training)
        list_scans_train = get_the_sequence_of_scans(
            csv_file = config['dataset']['csv_file'],
            subjects = subjects_train,
            age_min = config['dataset']['age_min'],
            age_max = config['dataset']['age_max'],
            mode = 'train',
            epochs = config['optimization']['epochs'],
        )
        list_scans_val = get_the_sequence_of_scans(
            csv_file = config['dataset']['csv_file'],
            subjects = subjects_val,
            age_min = config['dataset']['age_min'],
            age_max = config['dataset']['age_max'],
            mode = 'test',
            epochs = None,
        )
        dataloader_val = get_dataloader(
            modality=config.get('modality', 'DTI'),
            dataset_root = config['dataset']['root'],
            csv_file = config['dataset']['csv_file'],
            list_scans = list_scans_val,
            batch_size = config['optimization']['batch_size'],
        )

        # model
        model = get_the_resnet_model(
            model_name = config['model']['name'], 
            feature_vector_length = config['model']['feature_vector_length'], 
            MLP_hidden_layer_sizes = config['model']['mlp_hidden_layer_sizes'],
            n_input_channels=config['model']['n_input_channels'],
        ).to(device, non_blocking=True)
        
        # optimizer
        if config['optimization']['optimizer']['name'] == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr = config['optimization']['optimizer']['lr'],
            )
        else:
            raise ValueError("Optimizer not supported yet.")
        
        # learning rate scheduler
        if config['optimization']['lr_scheduler']['name'] == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer = optimizer,
                max_lr = config['optimization']['lr_scheduler']['lr_max'],
                epochs = len(list_scans_train),
                steps_per_epoch = math.ceil(len(list_scans_train[0])/(config['optimization']['batch_size']*config['optimization']['batch_size_mimic_factor'])),
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
        
        for epoch in range(len(list_scans_train)):
            
            # epoch-specific dataloader
            dataloader_train = get_dataloader(
                modality=config.get('modality', 'DTI'),
                dataset_root = config['dataset']['root'],
                csv_file = config['dataset']['csv_file'],
                list_scans = list_scans_train[epoch],
                batch_size = config['optimization']['batch_size'],
            )

            # training
            print(f"\nEpoch: {epoch+1}\nTraining phase:")
            model.train()
            epoch_loss = 0.0        
            optimizer.zero_grad()

            for i, (image, label_feature, age) in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
                image, label_feature, age = image.to(device, non_blocking=True), label_feature.to(device, non_blocking=True), age.to(device, non_blocking=True)
                
                with torch.autocast('cuda'):
                    output = model(image, label_feature)
                    loss = loss_fn(output, age.view(-1, 1))
                    epoch_loss += loss.detach()

                loss /= config['optimization']['batch_size_mimic_factor']
                loss.backward()

                if (i+1) % config['optimization']['batch_size_mimic_factor'] == 0 or (i+1) == len(dataloader_train):
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            
            epoch_loss /= len(dataloader_train)
            writer.add_scalar('Train/Loss', epoch_loss, (epoch+1))

            # QC PNG generation & garbage collection
            if (config['output']['png_sanity_check'] != '') and (PNG_GENERATED == False):
                generate_png_during_training(
                    img_tensor = image[0,:,:,:,:].cpu(), 
                    path_png = Path(config['output']['png_sanity_check']) / f"age-{age[0].cpu().numpy():.1f}.png"
                )
                print(f"Generated PNG for sanity check at {config['output']['png_sanity_check']}")
                PNG_GENERATED = True
            gc.collect()
            torch.cuda.empty_cache()

            # validation
            print("Validation phase:")
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for image, label_feature, age in tqdm(dataloader_val):
                    image, label_feature, age = image.to(device, non_blocking=True), label_feature.to(device, non_blocking=True), age.to(device, non_blocking=True)

                    output = model(image, label_feature)                        
                    val_loss += loss_fn(output, age.view(-1, 1)).detach()

                val_loss /= len(dataloader_val)
            writer.add_scalar('Val/Loss', val_loss, (epoch+1))
            print(f"Current LR: {scheduler.get_last_lr()}\tTrain Loss: {epoch_loss}\tValidation Loss: {val_loss}")
            torch.cuda.empty_cache()

            # save improved model to local
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                path_pth_local = weights_local_dir / f"{config['model']['name']}_fold-{fold_idx}_epoch-{epoch+1}_valloss-{best_val_loss:.4f}.pth"
                torch.save(model.state_dict(), path_pth_local)
                print(f'Saved improved model to {path_pth_local} at epoch {epoch+1} with validation loss {best_val_loss}')
                
        path_pth_server = weights_server_dir / path_pth_local.name
        print(f"Copying best model of fold-{fold_idx} to {path_pth_server}\n")
        subprocess.run(['cp', path_pth_local, path_pth_server])
        
        writer.close()