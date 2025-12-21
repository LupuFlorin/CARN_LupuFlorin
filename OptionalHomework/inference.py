# -*- coding: utf-8 -*-


import sys
!{sys.executable} -m pip install timed-decorator

import sys
!{sys.executable} -m pip install wandb

"""
Inference and benchmarking script with WandB integration
Tracks: inference times, speedups, GPU metrics, comparison plots
Perfect for the assignment's benchmarking requirement
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
import time
import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm


class ImprovedTransformNet(nn.Module):
    """Same architecture as training"""
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

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
        x3 = self.conv3(x2) + x2
        x4 = self.conv4(x3)
        out = self.output(x4)
        return self.pool(out)


def get_cifar10_images(data_path: str, train: bool):
    """Load CIFAR-10 images"""
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])
    cifar = CIFAR10(root=data_path, train=train,
                    transform=transforms, download=True)
    return [img for img, _ in cifar]


def get_ground_truth_transform():
    """Target transformation"""
    return v2.Compose([
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        v2.functional.hflip,
        v2.functional.vflip,
    ])


def benchmark_cpu_transforms(images, num_runs=3):
    """
    Benchmark sequential CPU transformations
    Run multiple times for accuracy
    """
    transforms = get_ground_truth_transform()
    times = []

    for run in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()

        results = [transforms(img) for img in images]

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        times.append(elapsed)

    return np.mean(times), np.std(times), results


@torch.no_grad()
def benchmark_model_inference(model, images, batch_size, device, num_runs=3):
    """
    Benchmark model inference with batching
    Run multiple times for accuracy
    """
    model.eval()
    times = []

    # Prepare data
    dataset = TensorDataset(torch.stack(images))
    dataloader = DataLoader(dataset, batch_size=batch_size,
                           shuffle=False, pin_memory=True)

    # Warmup run (important for GPU)
    for (batch,) in dataloader:
        batch = batch.to(device)
        _ = model(batch)

    # Actual benchmarking
    for run in range(num_runs):
        results = []

        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()

        for (batch,) in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            results.append(outputs.cpu())

        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = time.time() - start
        times.append(elapsed)

    return np.mean(times), np.std(times), torch.cat(results)


def test_single_configuration(model, images, device, batch_size, num_images=None):
    """
    Test single configuration and return detailed results
    """
    if num_images:
        images = images[:num_images]

    print(f"\n{'='*70}")
    print(f"Testing: device={device.type}, batch_size={batch_size}, n_images={len(images)}")
    print(f"{'='*70}")

    # Benchmark CPU
    print("Running CPU baseline (3 runs)...")
    cpu_time_mean, cpu_time_std, _ = benchmark_cpu_transforms(images)

    # Benchmark Model
    print(f"Running model on {device.type} (3 runs)...")
    model_time_mean, model_time_std, _ = benchmark_model_inference(
        model, images, batch_size, device
    )

    # Calculate metrics
    speedup = cpu_time_mean / model_time_mean
    throughput_cpu = len(images) / cpu_time_mean
    throughput_model = len(images) / model_time_mean

    results = {
        'device': device.type,
        'batch_size': batch_size,
        'num_images': len(images),
        'cpu_time_mean': cpu_time_mean,
        'cpu_time_std': cpu_time_std,
        'model_time_mean': model_time_mean,
        'model_time_std': model_time_std,
        'speedup': speedup,
        'throughput_cpu': throughput_cpu,
        'throughput_model': throughput_model,
        'faster_than_cpu': speedup > 1.0,
    }

    # Print results
    print(f"\n{'Results':^70}")
    print(f"{'-'*70}")
    print(f"CPU Sequential:     {cpu_time_mean:.4f}s ± {cpu_time_std:.4f}s")
    print(f"Model ({device.type}):        {model_time_mean:.4f}s ± {model_time_std:.4f}s")
    print(f"Speedup:            {speedup:.2f}x {'✓ FASTER' if speedup > 1 else '✗ SLOWER'}")
    print(f"Throughput (CPU):   {throughput_cpu:.1f} images/sec")
    print(f"Throughput (Model): {throughput_model:.1f} images/sec")
    print(f"{'='*70}\n")

    return results


def create_benchmark_plots(all_results, save_dir='./'):
    """Create comprehensive benchmark visualization"""
    df = pd.DataFrame(all_results)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Inference Time by Batch Size
    ax1 = fig.add_subplot(gs[0, :2])
    for device in df['device'].unique():
        device_data = df[df['device'] == device]
        ax1.errorbar(device_data['batch_size'], device_data['model_time_mean'],
                    yerr=device_data['model_time_std'], marker='o',
                    label=f'{device.upper()}', linewidth=2, markersize=8, capsize=5)

    cpu_baseline = df['cpu_time_mean'].iloc[0]
    ax1.axhline(cpu_baseline, color='red', linestyle='--',
               label='CPU Baseline', linewidth=2)
    ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inference Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Inference Time vs Batch Size', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)

    # Plot 2: Speedup by Batch Size
    ax2 = fig.add_subplot(gs[0, 2])
    for device in df['device'].unique():
        device_data = df[df['device'] == device]
        ax2.plot(device_data['batch_size'], device_data['speedup'],
                marker='o', label=f'{device.upper()}', linewidth=2, markersize=8)

    ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax2.fill_between(df['batch_size'].unique(), 1.0,
                     df.groupby('batch_size')['speedup'].max(),
                     alpha=0.2, color='green')
    ax2.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup (x)', fontsize=12, fontweight='bold')
    ax2.set_title('Speedup vs Batch Size', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)

    # Plot 3: Throughput Comparison
    ax3 = fig.add_subplot(gs[1, :])
    x = np.arange(len(df['batch_size'].unique()))
    width = 0.35

    for idx, device in enumerate(df['device'].unique()):
        device_data = df[df['device'] == device].sort_values('batch_size')
        offset = width * (idx - 0.5)
        ax3.bar(x + offset, device_data['throughput_model'], width,
               label=f'{device.upper()}', alpha=0.8)

    cpu_throughput = df['throughput_cpu'].iloc[0]
    ax3.axhline(cpu_throughput, color='red', linestyle='--',
               linewidth=2, label='CPU Baseline')

    ax3.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Throughput (images/sec)', fontsize=12, fontweight='bold')
    ax3.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df['batch_size'].unique())
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Performance Heatmap
    ax4 = fig.add_subplot(gs[2, 0])
    pivot = df.pivot_table(values='speedup', index='device',
                           columns='batch_size', aggfunc='first')
    im = ax4.imshow(pivot, cmap='RdYlGn', aspect='auto', vmin=0, vmax=pivot.max())
    ax4.set_xticks(np.arange(len(pivot.columns)))
    ax4.set_yticks(np.arange(len(pivot.index)))
    ax4.set_xticklabels(pivot.columns)
    ax4.set_yticklabels([idx.upper() for idx in pivot.index])
    ax4.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Device', fontsize=12, fontweight='bold')
    ax4.set_title('Speedup Heatmap', fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            text = ax4.text(j, i, f'{pivot.iloc[i, j]:.2f}x',
                          ha="center", va="center", color="black", fontsize=10)

    plt.colorbar(im, ax=ax4, label='Speedup')



    # Save figure
    save_path = f'{save_dir}/comprehensive_benchmark.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comprehensive plot to: {save_path}")
    plt.close()

    return fig, save_path


def comprehensive_benchmark_with_wandb(model_path='improved_transform_model_wandb.pth',
                                       num_test_images=10000):
    """
    Run comprehensive benchmarks with WandB tracking
    """
    print("="*70)
    print("COMPREHENSIVE BENCHMARK WITH WANDB")
    print("="*70)

    # Initialize WandB
    run = wandb.init(
        project="neural-image-transformation",
        name="benchmark-evaluation",
        job_type="inference",
        tags=["benchmark", "inference", "performance"],
        notes="Comprehensive benchmarking across devices and batch sizes"
    )

    # Load model
    print("\n1. Loading model...")
    model = ImprovedTransformNet()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Log model config
    if 'config' in checkpoint:
        wandb.config.update(checkpoint['config'])

    total_params = sum(p.numel() for p in model.parameters())
    wandb.config.update({
        'model_parameters': total_params,
        'num_test_images': num_test_images,
    })

    print(f"✓ Model loaded: {total_params:,} parameters")

    # Load test data
    print("\n2. Loading test data...")
    test_images = get_cifar10_images('./data', train=False)[:num_test_images]
    print(f"✓ Loaded {len(test_images)} test images")

    # Determine available devices
    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices.append(torch.device('cuda'))
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        devices.append(torch.device('mps'))
        print("✓ MPS (Apple Silicon) available")

    # Test configurations
    batch_sizes = [32, 64, 128, 256, 512]
    all_results = []

    # Run benchmarks
    print(f"\n3. Running benchmarks...")
    print(f"   Devices: {[d.type for d in devices]}")
    print(f"   Batch sizes: {batch_sizes}")

    for device in devices:
        model_device = model.to(device)

        for batch_size in batch_sizes:
            try:
                # Run benchmark
                results = test_single_configuration(
                    model_device, test_images, device, batch_size
                )
                all_results.append(results)

                # Log to WandB
                wandb.log({
                    f'{device.type}/time_bs_{batch_size}': results['model_time_mean'],
                    f'{device.type}/speedup_bs_{batch_size}': results['speedup'],
                    f'{device.type}/throughput_bs_{batch_size}': results['throughput_model'],
                })

            except RuntimeError as e:
                print(f"⚠ Skipping {device.type} batch_size={batch_size}: {e}")
                continue

    # Create results DataFrame
    df = pd.DataFrame(all_results)

    # Create comprehensive plots
    print("\n4. Creating visualizations...")
    fig, plot_path = create_benchmark_plots(all_results)

    # Log plot to WandB
    wandb.log({"comprehensive_benchmark": wandb.Image(plot_path)})

    # Create and log results table
    table = wandb.Table(dataframe=df)
    wandb.log({"benchmark_results_table": table})

    # Save results to CSV
    csv_path = './benchmark_results_detailed.csv'
    df.to_csv(csv_path, index=False)
    wandb.save(csv_path)
    print(f"✓ Saved results to: {csv_path}")

    # Log summary statistics
    print("\n5. Computing summary statistics...")
    for device_type in df['device'].unique():
        device_data = df[df['device'] == device_type]
        faster_configs = device_data[device_data['faster_than_cpu']]

        if len(faster_configs) > 0:
            best = faster_configs.loc[faster_configs['speedup'].idxmax()]

            wandb.run.summary[f'{device_type}_max_speedup'] = best['speedup']
            wandb.run.summary[f'{device_type}_best_batch_size'] = int(best['batch_size'])
            wandb.run.summary[f'{device_type}_best_time'] = best['model_time_mean']
            wandb.run.summary[f'{device_type}_num_faster_configs'] = len(faster_configs)

            print(f"\n{device_type.upper()} Summary:")
            print(f"  ✓ Max speedup: {best['speedup']:.2f}x")
            print(f"  ✓ Best batch size: {int(best['batch_size'])}")
            print(f"  ✓ Best time: {best['model_time_mean']:.4f}s")
            print(f"  ✓ Faster than CPU: {len(faster_configs)}/{len(device_data)} configs")

    # Create summary plot for WandB
    fig_summary, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Summary: Best configurations
    best_configs = []
    for device_type in df['device'].unique():
        device_data = df[df['device'] == device_type]
        best = device_data.loc[device_data['speedup'].idxmax()]
        best_configs.append(best)

    best_df = pd.DataFrame(best_configs)

    axes[0].bar([f"{row['device'].upper()}\nbs={int(row['batch_size'])}"
                 for _, row in best_df.iterrows()],
                best_df['speedup'], alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].axhline(1.0, color='red', linestyle='--', linewidth=2, label='Break-even')
    axes[0].set_ylabel('Speedup (x)', fontsize=12, fontweight='bold')
    axes[0].set_title('Best Speedup per Device', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Summary: Time comparison
    axes[1].bar(['CPU\nSequential'] +
                [f"{row['device'].upper()}\nbs={int(row['batch_size'])}"
                 for _, row in best_df.iterrows()],
                [df['cpu_time_mean'].iloc[0]] + best_df['model_time_mean'].tolist(),
                alpha=0.7, edgecolor='black', linewidth=2,
                color=['red'] + ['green' if s > 1 else 'orange'
                                for s in best_df['speedup']])
    axes[1].set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    axes[1].set_title('Inference Time: Best Configurations', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    summary_path = './benchmark_summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    wandb.log({"benchmark_summary": wandb.Image(summary_path)})
    print(f"✓ Saved summary plot to: {summary_path}")

    # Final summary
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print("="*70)
    print(f"\n✓ Results logged to WandB: {run.url}")
    print(f"✓ Comprehensive plot: {plot_path}")
    print(f"✓ Summary plot: {summary_path}")
    print(f"✓ Detailed CSV: {csv_path}")
    print("\nView your results at:", run.url)

    wandb.finish()

    return all_results, df


def quick_benchmark(model_path='improved_transform_model_wandb.pth',
                   batch_size=128, device=None):
    """Quick single-configuration benchmark with WandB"""

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize WandB
    wandb.init(
        project="neural-image-transformation",
        name=f"quick-benchmark-{device.type}-bs{batch_size}",
        job_type="inference",
        tags=["quick-test", "benchmark"],
        config={'batch_size': batch_size, 'device': str(device)}
    )

    # Load model
    model = ImprovedTransformNet()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Load data
    test_images = get_cifar10_images('./data', train=False)

    # Run benchmark
    results = test_single_configuration(model, test_images, device, batch_size)

    # Log to WandB
    wandb.log({
        'cpu_time': results['cpu_time_mean'],
        'model_time': results['model_time_mean'],
        'speedup': results['speedup'],
        'throughput': results['throughput_model'],
        'faster_than_cpu': results['faster_than_cpu'],
    })

    # Create simple comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.bar(['CPU Sequential', f'Model ({device.type})\nbs={batch_size}'],
           [results['cpu_time_mean'], results['model_time_mean']],
           yerr=[results['cpu_time_std'], results['model_time_std']],
           alpha=0.7, capsize=5, edgecolor='black', linewidth=2,
           color=['red', 'green' if results['speedup'] > 1 else 'orange'])

    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(f'Inference Time Comparison\nSpeedup: {results["speedup"]:.2f}x',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('quick_benchmark.png', dpi=150, bbox_inches='tight')
    wandb.log({"comparison": wandb.Image('quick_benchmark.png')})

    print(f"\n✓ Quick benchmark complete!")
    print(f"✓ View results: {wandb.run.url}")

    wandb.finish()


if __name__ == '__main__':
    # Run comprehensive benchmark
    print("Starting comprehensive benchmark with WandB tracking...\n")

    # Make sure you have the model file
    model_file = 'improved_transform_model_wandb.pth'

    try:
        results, df = comprehensive_benchmark_with_wandb(model_file)

        print("\n" + "="*70)
        print("SUCCESS! All benchmarks completed.")
        print("="*70)

    except FileNotFoundError:
        print(f"\n⚠ Error: Model file '{model_file}' not found!")
        print("Please run the training script first to generate the model.")
        print("\nAlternatively, try quick benchmark with a different model:")
        print("  python inference.py --quick --model your_model.pth")

    except Exception as e:
        print(f"\n⚠ Error during benchmarking: {e}")
        print("Check that all dependencies are installed and data is available.")