# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class ImprovedTransformNet(nn.Module):
    """
    Improved architecture with residual connections and better feature extraction
    Total params: ~50k (lightweight but effective)
    """
    def __init__(self):
        super().__init__()

        # Encoder with skip connections
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Downsample block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Middle processing
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Final output
        self.output = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.pool = nn.AdaptiveAvgPool2d((28, 28))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2) + x2  # Residual connection
        x4 = self.conv4(x3)
        out = self.output(x4)
        out = self.pool(out)
        return out


class MixedLoss(nn.Module):
    """Combined MSE + L1 loss for better detail preservation"""
    def __init__(self, mse_weight=0.7, l1_weight=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        l1_loss = self.l1(pred, target)
        total_loss = self.mse_weight * mse_loss + self.l1_weight * l1_loss

        # Return components for logging
        return total_loss, mse_loss.item(), l1_loss.item()


def get_cifar10_images(data_path: str, train: bool):
    initial_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    cifar_10 = CIFAR10(root=data_path, train=train,
                       transform=initial_transforms, download=True)
    return [image for image, _ in cifar_10]


def get_ground_truth_transform():
    return v2.Compose([
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        v2.functional.hflip,
        v2.functional.vflip,
    ])


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        target = self.transform(img)
        return img, target


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.00005, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f'Val loss improved: {self.best_loss:.6f} -> {val_loss:.6f}')
            self.best_loss = val_loss
            self.best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0


def log_predictions_to_wandb(model, test_images, transform, device, epoch, num_samples=5):
    """Log sample predictions to WandB"""
    model.eval()

    images_to_log = []

    with torch.no_grad():
        for i in range(num_samples):
            img = test_images[i].unsqueeze(0).to(device)
            pred = model(img).cpu().squeeze().numpy()
            target = transform(test_images[i]).squeeze().numpy()
            original = test_images[i].permute(1, 2, 0).numpy()

            # Calculate error
            mae = np.abs(pred - target).mean()

            # Create comparison image
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(original)
            axes[0].set_title('Original (3x32x32)')
            axes[0].axis('off')

            axes[1].imshow(pred, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title(f'Prediction (MAE: {mae:.4f})')
            axes[1].axis('off')

            axes[2].imshow(target, cmap='gray', vmin=0, vmax=1)
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')

            plt.tight_layout()

            # Log to wandb
            images_to_log.append(wandb.Image(fig, caption=f"Sample {i+1}"))
            plt.close(fig)

    wandb.log({"predictions": images_to_log, "epoch": epoch})


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch with WandB logging"""
    model.train()
    total_loss = 0
    total_mse = 0
    total_l1 = 0
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss, mse_loss, l1_loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        total_mse += mse_loss
        total_l1 += l1_loss

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        # Log batch metrics every 50 batches
        if batch_idx % 50 == 0:
            wandb.log({
                'batch_loss': loss.item(),
                'batch_mse': mse_loss,
                'batch_l1': l1_loss,
                'batch': epoch * num_batches + batch_idx
            })

    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    avg_l1 = total_l1 / num_batches

    return avg_loss, avg_mse, avg_l1


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate the model with WandB logging"""
    model.eval()
    total_loss = 0
    total_mse = 0
    total_l1 = 0
    num_batches = len(dataloader)

    for inputs, targets in tqdm(dataloader, desc='Validation', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss, mse_loss, l1_loss = criterion(outputs, targets)

        total_loss += loss.item()
        total_mse += mse_loss
        total_l1 += l1_loss

    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    avg_l1 = total_l1 / num_batches

    return avg_loss, avg_mse, avg_l1


def train_model(model, train_loader, val_loader, test_images, transform,
                criterion, optimizer, scheduler, num_epochs, device, early_stopping):
    """Complete training loop with WandB logging"""

    print(f"\n{'='*60}")
    print(f"Training with WandB Logging")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):
        # Training
        train_loss, train_mse, train_l1 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validation
        val_loss, val_mse, val_l1 = validate(model, val_loader, criterion, device)

        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics to WandB
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'train/mse': train_mse,
            'train/l1': train_l1,
            'val/loss': val_loss,
            'val/mse': val_mse,
            'val/l1': val_l1,
            'learning_rate': current_lr,
            'early_stopping_counter': early_stopping.counter,
        })

        print(f'\nEpoch {epoch}/{num_epochs}')
        print(f'Train Loss: {train_loss:.6f} (MSE: {train_mse:.6f}, L1: {train_l1:.6f})')
        print(f'Val Loss:   {val_loss:.6f} (MSE: {val_mse:.6f}, L1: {val_l1:.6f})')
        print(f'LR: {current_lr:.6f}')

        # Log predictions every 5 epochs
        if epoch % 5 == 0:
            log_predictions_to_wandb(model, test_images, transform, device, epoch)

        # Step scheduler
        scheduler.step(val_loss)

        # Early stopping
        early_stopping(val_loss, model)

        # Log best model to WandB
        if val_loss <= early_stopping.best_loss:
            wandb.run.summary["best_val_loss"] = val_loss
            wandb.run.summary["best_epoch"] = epoch

        if early_stopping.early_stop:
            print(f'\n🛑 Early stopping triggered at epoch {epoch}')
            model.load_state_dict(early_stopping.best_model)

            # Log final predictions
            log_predictions_to_wandb(model, test_images, transform, device, epoch)
            break

    return model

def main():
    # ============================================================
    # Configuration
    # ============================================================
    config = {
        'architecture': 'ImprovedTransformNet',
        'batch_size': 128,
        'num_epochs': 50,
        'initial_lr': 0.001,
        'optimizer': 'AdamW',
        'weight_decay': 1e-5,
        'loss_function': 'Mixed (MSE + L1)',
        'mse_weight': 0.7,
        'l1_weight': 0.3,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.00005,
        'scheduler': 'ReduceLROnPlateau',
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,
        'gradient_clip_norm': 1.0,
        'val_split': 0.1,
        'device': str(device),
    }

    # ============================================================
    # Initialize WandB
    # ============================================================
    wandb.init(
        project="neural-image-transformation",  # Change this to your project name
        name="improved-transformnet-run",        # Run name
        config=config,
        tags=["cifar10", "image-transformation", "cnn"],
        notes="Training improved architecture with mixed loss and WandB logging"
    )

    # ============================================================
    # Load Data
    # ============================================================
    print("Loading CIFAR-10...")
    train_images = get_cifar10_images('./data', train=True)
    test_images = get_cifar10_images('./data', train=False)

    # Create datasets
    transform = get_ground_truth_transform()
    full_dataset = TransformDataset(train_images, transform)

    val_size = int(len(full_dataset) * config['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                             shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           shuffle=False, num_workers=2, pin_memory=True)

    # ============================================================
    # Initialize Model
    # ============================================================
    model = ImprovedTransformNet().to(device)

    # Log model architecture
    wandb.watch(model, log='all', log_freq=100)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    config['total_parameters'] = total_params
    config['trainable_parameters'] = trainable_params
    wandb.config.update(config)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ============================================================
    # Training Components
    # ============================================================
    criterion = MixedLoss(
        mse_weight=config['mse_weight'],
        l1_weight=config['l1_weight']
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['initial_lr'],
        weight_decay=config['weight_decay']
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['scheduler_factor'],
        patience=config['scheduler_patience']
    )

    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        min_delta=config['early_stopping_min_delta'],
        verbose=True
    )

    # ============================================================
    # Train Model
    # ============================================================
    model = train_model(
        model, train_loader, val_loader, test_images, transform,
        criterion, optimizer, scheduler, config['num_epochs'],
        device, early_stopping
    )

    # ============================================================
    # Final Evaluation
    # ============================================================
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)

    # Final predictions
    log_predictions_to_wandb(model, test_images, transform, device, epoch=-1, num_samples=10)

    # Save model locally
    model_path = 'improved_transform_model_wandb.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_val_loss': early_stopping.best_loss,
    }, model_path)

    # Save model to WandB
    wandb.save(model_path)

    print(f"\n✅ Training complete!")
    print(f"✅ Best validation loss: {early_stopping.best_loss:.6f}")
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ View results at: {wandb.run.url}")

    # Finish WandB run
    wandb.finish()


if __name__ == '__main__':
    main()