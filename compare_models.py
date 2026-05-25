"""
Compare LipNet-GRU vs LipNet-Transformer after training.

Usage:
    python compare_models.py \\
        --gru_ckpt   checkpoints/gru/gru_best.pt \\
        --gru_hist   checkpoints/gru/history.json \\
        --trans_ckpt checkpoints/transformer/transformer_best.pt \\
        --trans_hist checkpoints/transformer/history.json \\
        --video_path /data/grid/lip \\
        --anno_path  /data/grid/GRID_align_txt \\
        --val_list   data/overlap_val.txt \\
        --out_dir    results/
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from jiwer import wer as jiwer_wer, cer as jiwer_cer

from dataset import MyDataset, ctc_beam_decode  # [v2] beam search
from models import LipNetGRU, LipNetTransformer


def _get_device(requested: str) -> torch.device:
    if requested == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(requested)


def _synchronize(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# [v2] beam search decode (substituí greedy)
def ctc_decode(y: torch.Tensor) -> list:
    """CTC beam search decode. y: (B, T, C)."""
    log_probs = y.log_softmax(-1).cpu()
    return [ctc_beam_decode(log_probs[i]) for i in range(y.size(0))]


def load_model(model_type: str, ckpt_path: str, args) -> nn.Module:
    if model_type == 'gru':
        model = LipNetGRU(num_classes=args.num_classes)
    else:
        model = LipNetTransformer(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            num_classes=args.num_classes,
        )
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, args, device: torch.device) -> dict:
    """Run inference on the validation set; return WER, CER, avg loss."""
    model = model.to(device)
    dataset = MyDataset(
        args.video_path, args.anno_path, args.val_list,
        args.vid_padding, args.txt_padding, 'test',
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
    )
    crit = nn.CTCLoss(zero_infinity=True)

    all_preds, all_truths = [], []
    losses = []

    nb = device.type == 'cuda'
    with torch.no_grad():
        for batch in loader:
            vid = batch['vid'].to(device, non_blocking=nb)
            txt = batch['txt'].to(device, non_blocking=nb)
            vid_len = batch['vid_len'].to(device, non_blocking=nb)
            txt_len = batch['txt_len'].to(device, non_blocking=nb)

            y = model(vid)
            loss = crit(
                y.transpose(0, 1).log_softmax(-1),
                txt, vid_len.view(-1), txt_len.view(-1),
            )
            losses.append(loss.item())

            preds = ctc_decode(y)
            truths = [MyDataset.arr2txt(txt[i], start=1) for i in range(txt.size(0))]
            all_preds.extend(preds)
            all_truths.extend(truths)

    # jiwer expects non-empty strings; replace empty with a single space
    preds_safe = [p if p.strip() else ' ' for p in all_preds]
    truths_safe = [t if t.strip() else ' ' for t in all_truths]

    return {
        'loss': float(np.mean(losses)),
        'wer': jiwer_wer(truths_safe, preds_safe),
        'cer': jiwer_cer(truths_safe, preds_safe),
    }


# ---------------------------------------------------------------------------
# Inference latency benchmark
# ---------------------------------------------------------------------------

def benchmark_inference(model: nn.Module, args, device: torch.device,
                        n_warmup: int = 10, n_runs: int = 50) -> float:
    """Return mean inference time (seconds) per single sample."""
    model = model.to(device).eval()
    dummy = torch.randn(1, 3, args.vid_padding, 64, 128).to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
        _synchronize(device)

        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(dummy)
        _synchronize(device)

    return (time.perf_counter() - t0) / n_runs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_curves(gru_hist: dict, trans_hist: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    metrics = [
        ('train_loss', 'Training Loss'),
        ('val_loss',   'Validation Loss'),
        ('val_wer',    'Validation WER'),
        ('val_cer',    'Validation CER'),
    ]

    # --- Combined 4-panel figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LipNet-GRU vs LipNet-Transformer', fontsize=14, fontweight='bold')

    for ax, (key, title) in zip(axes.flat, metrics):
        gru_epochs = gru_hist.get('epochs', list(range(1, len(gru_hist[key]) + 1)))
        tr_epochs  = trans_hist.get('epochs', list(range(1, len(trans_hist[key]) + 1)))

        ax.plot(gru_epochs, gru_hist[key], label='GRU', color='steelblue', linewidth=1.8)
        ax.plot(tr_epochs, trans_hist[key], label='Transformer',
                color='darkorange', linewidth=1.8, linestyle='--')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'training_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')

    # --- Separate loss-only figure for paper/slides ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    gru_epochs = gru_hist.get('epochs', list(range(1, len(gru_hist['val_loss']) + 1)))
    tr_epochs  = trans_hist.get('epochs', list(range(1, len(trans_hist['val_loss']) + 1)))
    ax2.plot(gru_epochs, gru_hist['train_loss'],
             color='steelblue', linewidth=1.5, alpha=0.5, linestyle=':')
    ax2.plot(gru_epochs, gru_hist['val_loss'],
             color='steelblue', linewidth=2.0, label='GRU (val)')
    ax2.plot(tr_epochs, trans_hist['train_loss'],
             color='darkorange', linewidth=1.5, alpha=0.5, linestyle=':')
    ax2.plot(tr_epochs, trans_hist['val_loss'],
             color='darkorange', linewidth=2.0, linestyle='--', label='Transformer (val)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('CTC Loss')
    ax2.set_title('Train (dotted) vs Validation (solid) Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    path2 = os.path.join(out_dir, 'loss_curves.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved: {path2}')

    # --- WER comparison ---
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(gru_epochs, gru_hist['val_wer'],
             color='steelblue', linewidth=2.0, label='GRU')
    ax3.plot(tr_epochs, trans_hist['val_wer'],
             color='darkorange', linewidth=2.0, linestyle='--', label='Transformer')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('WER')
    ax3.set_title('Validation Word Error Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    path3 = os.path.join(out_dir, 'wer_curves.png')
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f'Saved: {path3}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print('Loading models...')
    gru_model   = load_model('gru',         args.gru_ckpt,   args)
    trans_model = load_model('transformer', args.trans_ckpt, args)

    # Parameter counts
    gru_params   = count_parameters(gru_model)
    trans_params = count_parameters(trans_model)

    device = _get_device(args.device)

    # Inference latency
    print('Benchmarking inference speed...')
    gru_latency   = benchmark_inference(gru_model,   args, device)
    trans_latency = benchmark_inference(trans_model, args, device)

    # WER / CER via jiwer
    print('Evaluating GRU on validation set...')
    gru_metrics = evaluate(gru_model, args, device)
    print('Evaluating Transformer on validation set...')
    trans_metrics = evaluate(trans_model, args, device)

    # Load training histories
    with open(args.gru_hist) as f:
        gru_hist = json.load(f)
    with open(args.trans_hist) as f:
        trans_hist = json.load(f)

    # Plot
    plot_curves(gru_hist, trans_hist, args.out_dir)

    # Summary table
    sep = '=' * 60
    print(f'\n{sep}')
    print(f'{"Metric":<30} {"GRU":>12} {"Transformer":>14}')
    print(sep)
    print(f'{"Parameters":<30} {gru_params:>12,} {trans_params:>14,}')
    print(f'{"Inference time (ms)":<30} {gru_latency*1000:>12.2f} {trans_latency*1000:>14.2f}')
    print(f'{"Val Loss":<30} {gru_metrics["loss"]:>12.4f} {trans_metrics["loss"]:>14.4f}')
    print(f'{"WER (jiwer)":<30} {gru_metrics["wer"]:>12.4f} {trans_metrics["wer"]:>14.4f}')
    print(f'{"CER (jiwer)":<30} {gru_metrics["cer"]:>12.4f} {trans_metrics["cer"]:>14.4f}')
    print(sep)

    # Save summary to JSON
    summary = {
        'gru': {
            'params': gru_params,
            'inference_ms': round(gru_latency * 1000, 3),
            **gru_metrics,
        },
        'transformer': {
            'params': trans_params,
            'inference_ms': round(trans_latency * 1000, 3),
            **trans_metrics,
        },
    }
    summary_path = os.path.join(args.out_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nFull summary saved to: {summary_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # Checkpoint paths
    p.add_argument('--gru_ckpt',   required=True, help='Path to GRU best checkpoint (.pt)')
    p.add_argument('--gru_hist',   required=True, help='Path to GRU history.json')
    p.add_argument('--trans_ckpt', required=True, help='Path to Transformer best checkpoint (.pt)')
    p.add_argument('--trans_hist', required=True, help='Path to Transformer history.json')

    # Dataset
    p.add_argument('--video_path', default='/data/grid/lip')
    p.add_argument('--anno_path',  default='/data/grid/GRID_align_txt')
    p.add_argument('--val_list',   default='data/overlap_val.txt')
    p.add_argument('--vid_padding',  type=int, default=75)
    p.add_argument('--txt_padding',  type=int, default=200)
    p.add_argument('--num_classes',  type=int, default=28)
    p.add_argument('--batch_size',   type=int, default=8)
    p.add_argument('--num_workers',  type=int, default=4)

    # Transformer architecture (must match the trained model)
    p.add_argument('--d_model',        type=int, default=512)
    p.add_argument('--nhead',          type=int, default=8)
    p.add_argument('--num_layers',     type=int, default=2)
    p.add_argument('--dim_feedforward', type=int, default=2048)

    p.add_argument('--out_dir', default='results', help='Where to save PNGs and summary.json')
    p.add_argument('--gpu', default='0')
    p.add_argument('--device', default='auto',
                   help='Device: auto (cuda→mps→cpu), cuda, mps, or cpu')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.device in ('cuda', 'auto'):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
