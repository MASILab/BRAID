import os
import pdb
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from torch.utils.data import Dataset, DataLoader
from braid.models import get_the_resnet_model
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    MaskIntensityd,
    NormalizeIntensityd,
    ToTensord,
    Spacingd,
    ResizeWithPadOrCropd,
)
torch.set_flush_denormal(True)
os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536" # to address monotonic increase in memory usage
device = torch.device('cuda')
raw_dataset_csv = {
    'train': '/fs5/p_masi/liuy140/share/chenyu/normal_train.csv',
    'val': '/fs5/p_masi/liuy140/share/chenyu/normal_val.csv',
    'test': '/fs5/p_masi/liuy140/share/chenyu/normal_test.csv',
}

def copy_file(job):
    path_src, path_dst = job
    if not os.path.exists(path_dst):
        os.makedirs(os.path.dirname(path_dst), exist_ok=True)
        os.system(f"rsync -rL {path_src} {path_dst}")

def prepare_local_dataset(dataset_local_root):
    raw_paths = []
    for raw_csv in raw_dataset_csv.values():
        df_raw = pd.read_csv(raw_csv)
        raw_paths += df_raw['t1w_path'].tolist()
        raw_paths += df_raw['mask_path'].tolist()
    raw_root = os.path.commonpath(raw_paths)

    copy_jobs = []
    local_dataset_csv = {}
    for split, raw_csv in raw_dataset_csv.items():
        local_csv = os.path.join(dataset_local_root, os.path.basename(raw_csv))
        
        df = pd.read_csv(raw_csv)
        df['raw_t1w_path'] = df['t1w_path']
        df['raw_mask_path'] = df['mask_path']

        df['t1w_path'] = df['t1w_path'].apply(
            lambda p: os.path.join(dataset_local_root, os.path.relpath(p, raw_root))
        )
        df['mask_path'] = df['mask_path'].apply(
            lambda p: os.path.join(dataset_local_root, os.path.relpath(p, raw_root))
        )
        df.to_csv(local_csv, index=False)
        local_dataset_csv[split] = local_csv

        for _, row in df.iterrows():
            copy_jobs.append((row['raw_t1w_path'], row['t1w_path']))
            copy_jobs.append((row['raw_mask_path'], row['mask_path']))
    
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(copy_file, copy_jobs), total=len(copy_jobs), desc="Copy data to local"))
    
    return local_dataset_csv


def get_model(args):
    # model configs
    model_name = 'resnet101'
    MLP_hidden_layer_sizes = []
    n_input_channels = 1

    if args.model_name == 'braid_t1w':
        feature_vector_length = 2  # sex
    elif args.model_name == 'resnet3d':
        feature_vector_length = 0
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")
    
    model = get_the_resnet_model(
        model_name=model_name, 
        feature_vector_length=feature_vector_length, 
        MLP_hidden_layer_sizes=MLP_hidden_layer_sizes, 
        n_input_channels=n_input_channels,
    )
    model = model.to(device)
    
    # load weights if testing
    if args.stage == 'test':
        assert os.path.exists(args.test_model_pth), "Model weights path must be provided for testing."
        checkpoint = torch.load(args.test_model_pth, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Model weights loaded from {args.test_model_pth}")
    return model

class T1wAgeDataset(Dataset):
    def __init__(self, path_csv, tabular=True):
        self.df = pd.read_csv(path_csv)
        self.tabular = tabular
        self.transforms = Compose([
            LoadImaged(keys=["t1w", "mask"], image_only=True),
            EnsureChannelFirstd(keys=["t1w", "mask"]),
            Orientationd(keys=["t1w", "mask"], axcodes="RAS"),
            MaskIntensityd(keys=["t1w"], mask_key="mask"),
            Spacingd(keys=["t1w"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
            ResizeWithPadOrCropd(keys=["t1w"], spatial_size=(128, 160, 128), mode="constant"),
            NormalizeIntensityd(keys=["t1w"], nonzero=True),
            ToTensord(keys=["t1w"])
        ])

    def __len__(self):
        return len(self.df.index)
    
    def sex2vector(self, sex):
        if sex == 'male':
            return torch.tensor([0.0, 1.0], dtype=torch.float32)
        elif sex == 'female':
            return torch.tensor([1.0, 0.0], dtype=torch.float32)
        elif sex == '':
            return torch.tensor([0.5, 0.5], dtype=torch.float32)
        else:
            raise ValueError(f"Unexpected sex label: {sex}")
    
    def __getitem__(self, idx):
        sample = {
            "t1w": self.df.iloc[idx]['t1w_path'],
            "mask": self.df.iloc[idx]['mask_path']
        }
        sample = self.transforms(sample)
        img = sample["t1w"]
    
        label_feature = self.sex2vector(self.df.iloc[idx]['sex']) if self.tabular else None
        age = torch.tensor(self.df.iloc[idx]['age'], dtype=torch.float32)
        return img, label_feature, age    

def train_model(model, dl_train, dl_val, epochs, args):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=1e-3,
        epochs=epochs,
        steps_per_epoch=len(dl_train),
    )
    loss_fn = torch.nn.MSELoss()

    tb_dir = os.path.join(args.output_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    best_val_loss = float('inf')

    weights_dir = os.path.join(args.output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for img, label_feature, age in tqdm(dl_train, total=len(dl_train), desc=f"Epoch: {epoch}"):
            optimizer.zero_grad()
            img = img.to(device, non_blocking=True)
            label_feature = label_feature.to(device, non_blocking=True)
            age = age.to(device, non_blocking=True)

            with torch.autocast('cuda'):
                output = model(img, label_feature)
                loss = loss_fn(output, age.view(-1, 1))
            
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
        
        epoch_loss /= len(dl_train)
        writer.add_scalar('Train/Loss', epoch_loss, epoch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, label_feature, age in tqdm(dl_val, total=len(dl_val), desc="Validation"):
                img, label_feature, age = img.to(device, non_blocking=True), label_feature.to(device, non_blocking=True), age.to(device, non_blocking=True)

                output = model(img, label_feature)
                val_loss += loss_fn(output, age.view(-1, 1)).detach()
            val_loss /= len(dl_val)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        print(f"Epoch {epoch}: Train Loss: {epoch_loss}\tValidation Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path_pth = os.path.join(weights_dir, f"epoch-{epoch}_val-loss-{best_val_loss:.4f}.pth")
            torch.save(model.state_dict(), path_pth)

    writer.close()


def test_model(model, dl_test, path_csv, args):
    output_csv = os.path.join(args.output_dir, "prediction.csv")
    if os.path.exists(output_csv):
        print(f"Output CSV already exists: {output_csv}")
        return
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df = pd.read_csv(path_csv)

    gt_all = []
    pred_all = []
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        for img, label_feature, age in tqdm(dl_test, total=len(dl_test), desc="Test"):
            img, label_feature = img.to(device, non_blocking=True), label_feature.to(device, non_blocking=True)

            gt = age
            pred = model(img, label_feature).detach().cpu()
            gt_all += torch.flatten(gt).tolist()
            pred_all += torch.flatten(pred).tolist()
    df['age_gt'] = gt_all
    df['age_pred'] = pred_all
    assert np.isclose(df['age'].values, df['age_gt'].values).all(), "Ground truth ages do not match!"
    df.to_csv(output_csv, index=False)


def main(args):
    local_dataset_csv = prepare_local_dataset(args.dataset_local_root)
    if args.model_name == "braid_t1w":
        tabular = True
    elif args.model_name == "resnet3d":
        tabular = False
    
    model = get_model(args)
    if args.stage == 'train':
        ds_train = T1wAgeDataset(local_dataset_csv['train'], tabular=tabular)
        ds_val = T1wAgeDataset(local_dataset_csv['val'], tabular=tabular)
        dl_train = DataLoader(
            ds_train,
            batch_size=4,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=4,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
        )
        train_model(
            model=model, 
            dl_train=dl_train, 
            dl_val=dl_val, 
            epochs=40, 
            args=args
        )
    elif args.stage == 'test':
        ds_test = T1wAgeDataset(local_dataset_csv['test'], tabular=tabular)
        dl_test = DataLoader(
            ds_test,
            batch_size=4,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
        )
        test_model(model, dl_test, local_dataset_csv['test'], args)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark 3D ResNet and BRAID T1w on MetaVoxel datasets.")
    parser.add_argument(
        "--model_name", type=str, default="braid_t1w", choices=['braid_t1w', 'resnet3d'],
        help="Name of the model to benchmark."
    )
    parser.add_argument(
        "--stage", type=str, default="train", choices=["train", "test"],
        help="Stage: train or test."
    )
    parser.add_argument(
        "--test_model_pth", type=str, default="",
        help="Path to model weights (.pth) for testing. Required if stage is 'test'."
    )
    parser.add_argument(
        "--dataset_local_root", type=str, default="/home-local/gaoc11/metavoxel",
        help="Local directory to store the datasets. Preferred to be an SSD for speed."
    )
    parser.add_argument(
        "--output_dir", type=str, default="/nfs/masi/gaoc11/projects/BRAID/experiments/2025-09-23_Yihao/outputs",
        help="Directory to save the outputs."
    )
    args = parser.parse_args()
    
    main(args)
