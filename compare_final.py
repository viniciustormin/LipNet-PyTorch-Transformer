"""
Comparação final dos 3 modelos: LipNet-GRU, Transformer v1 e Transformer v2.

Uso:
    python compare_final.py \\
        --gru_ckpt    checkpoints/s1/gru/gru_best.pt \\
        --gru_hist    checkpoints/s1/gru/history.json \\
        --v1_ckpt     checkpoints/s1/transformer/transformer_best.pt \\
        --v1_hist     checkpoints/s1/transformer/history.json \\
        --v2_ckpt     checkpoints/s1/transformer_v2/transformer_best.pt \\
        --v2_hist     checkpoints/s1/transformer_v2/history.json \\
        --video_path  data/grid/lip \\
        --anno_path   data/grid/GRID_align_txt \\
        --val_list    data/active_val.txt \\
        --out_dir     results/s1

Curvas de treino: lidas dos history.json (greedy WER para GRU/v1, beam WER para v2).
Tabela final: todos os 3 modelos re-avaliados com beam search para comparação justa.
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


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# [v2] beam search decode global
def ctc_decode(y: torch.Tensor) -> list:
    log_probs = y.log_softmax(-1).cpu()
    return [ctc_beam_decode(log_probs[i]) for i in range(y.size(0))]


def convergence_epoch(val_loss: list, threshold: float = 0.01, window: int = 3) -> int:
    """Primeira época em que val_loss não cai mais que `threshold` por `window` épocas."""
    for i in range(len(val_loss) - window):
        drops = [val_loss[i + j] - val_loss[i + j + 1] for j in range(window)]
        if all(d < threshold for d in drops):
            return i + 1
    return len(val_loss)  # nunca convergiu dentro do critério


# ---------------------------------------------------------------------------
# Carregamento de modelos
# ---------------------------------------------------------------------------

def load_gru(ckpt_path: str, args) -> nn.Module:
    model = LipNetGRU(num_classes=args.num_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    return model.eval()


def load_transformer(ckpt_path: str, args) -> nn.Module:
    model = LipNetTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        num_classes=args.num_classes,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    return model.eval()


# ---------------------------------------------------------------------------
# Avaliação com beam search (comparação justa entre os 3 modelos)
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, args, device: torch.device) -> dict:
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

    all_preds, all_truths, losses = [], [], []
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

    preds_safe  = [p if p.strip() else ' ' for p in all_preds]
    truths_safe = [t if t.strip() else ' ' for t in all_truths]
    return {
        'loss': float(np.mean(losses)),
        'wer':  jiwer_wer(truths_safe, preds_safe),
        'cer':  jiwer_cer(truths_safe, preds_safe),
    }


def benchmark_inference(model: nn.Module, args, device: torch.device,
                        n_warmup: int = 5, n_runs: int = 20) -> float:
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
# Plotagem
# ---------------------------------------------------------------------------

def plot_curves(hists: list[dict], labels: list[str], colors: list[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    metrics = [
        ('train_loss', 'Training Loss'),
        ('val_loss',   'Validation Loss'),
        ('val_wer',    'Validation WER'),
        ('val_cer',    'Validation CER'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LipNet — GRU vs Transformer v1 vs Transformer v2', fontsize=14, fontweight='bold')

    note = ('Curvas de treino: GRU e Transformer v1 avaliados com greedy decode; '
            'Transformer v2 com beam search (width=10)')
    fig.text(0.5, 0.01, note, ha='center', fontsize=8, color='gray', style='italic')

    for ax, (key, title) in zip(axes.flat, metrics):
        for hist, label, color in zip(hists, labels, colors):
            epochs = hist.get('epochs', list(range(1, len(hist[key]) + 1)))
            ax.plot(epochs, hist[key], label=label, color=color, linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(out_dir, 'comparison_final.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = _get_device(args.device)
    print(f'Device: {device}')

    # Carrega históricos de treino
    with open(args.gru_hist) as f:
        gru_hist = json.load(f)
    with open(args.v1_hist) as f:
        v1_hist = json.load(f)
    with open(args.v2_hist) as f:
        v2_hist = json.load(f)

    # Gera gráficos a partir dos históricos
    plot_curves(
        hists=[gru_hist, v1_hist, v2_hist],
        labels=['GRU', 'Transformer v1', 'Transformer v2'],
        colors=['steelblue', 'darkorange', 'seagreen'],
        out_dir=args.out_dir,
    )

    # Carrega modelos e re-avalia com beam search (comparação justa)
    print('\nCarregando modelos...')
    gru_model = load_gru(args.gru_ckpt, args)
    v1_model  = load_transformer(args.v1_ckpt, args)
    v2_model  = load_transformer(args.v2_ckpt, args)

    print('Benchmarking inferência...')
    gru_lat = benchmark_inference(gru_model, args, device)
    v1_lat  = benchmark_inference(v1_model,  args, device)
    v2_lat  = benchmark_inference(v2_model,  args, device)

    print('Avaliando GRU (beam search)...')
    gru_metrics = evaluate(gru_model, args, device)
    print('Avaliando Transformer v1 (beam search)...')
    v1_metrics  = evaluate(v1_model,  args, device)
    print('Avaliando Transformer v2 (beam search)...')
    v2_metrics  = evaluate(v2_model,  args, device)

    # Épocas de convergência
    gru_conv = convergence_epoch(gru_hist['val_loss'])
    v1_conv  = convergence_epoch(v1_hist['val_loss'])
    v2_conv  = convergence_epoch(v2_hist['val_loss'])

    # Parâmetros
    gru_params = count_parameters(gru_model)
    v1_params  = count_parameters(v1_model)
    v2_params  = count_parameters(v2_model)

    # Tabela
    sep = '=' * 80
    fmt = '{:<22} {:>10} {:>10} {:>22} {:>12}'
    print(f'\n{sep}')
    print(fmt.format('Modelo', 'WER (beam)', 'CER (beam)', 'Época convergência', 'Params'))
    print(sep)
    print(fmt.format('LipNet-GRU',        f'{gru_metrics["wer"]:.4f}', f'{gru_metrics["cer"]:.4f}', str(gru_conv), f'{gru_params:,}'))
    print(fmt.format('LipNet-Transf v1',  f'{v1_metrics["wer"]:.4f}',  f'{v1_metrics["cer"]:.4f}',  str(v1_conv),  f'{v1_params:,}'))
    print(fmt.format('LipNet-Transf v2',  f'{v2_metrics["wer"]:.4f}',  f'{v2_metrics["cer"]:.4f}',  str(v2_conv),  f'{v2_params:,}'))
    print(sep)
    print('* WER/CER re-avaliados com beam search (width=10) para comparação justa.')
    print(f'* Época convergência: 1ª época onde val_loss cai < 0.01 por 3 épocas consecutivas.')

    # Salva summary_final.json
    summary = {
        'gru':            {'params': gru_params, 'inference_ms': round(gru_lat*1000, 3), 'convergence_epoch': gru_conv, **gru_metrics},
        'transformer_v1': {'params': v1_params,  'inference_ms': round(v1_lat*1000, 3),  'convergence_epoch': v1_conv,  **v1_metrics},
        'transformer_v2': {'params': v2_params,  'inference_ms': round(v2_lat*1000, 3),  'convergence_epoch': v2_conv,  **v2_metrics},
    }
    summary_path = os.path.join(args.out_dir, 'summary_final.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSalvo: {summary_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Comparação final: GRU vs Transformer v1 vs v2')

    # Checkpoints e históricos
    p.add_argument('--gru_ckpt',  required=True)
    p.add_argument('--gru_hist',  required=True)
    p.add_argument('--v1_ckpt',   required=True)
    p.add_argument('--v1_hist',   required=True)
    p.add_argument('--v2_ckpt',   required=True)
    p.add_argument('--v2_hist',   required=True)

    # Dataset
    p.add_argument('--video_path',  default='data/grid/lip')
    p.add_argument('--anno_path',   default='data/grid/GRID_align_txt')
    p.add_argument('--val_list',    default='data/overlap_val.txt')
    p.add_argument('--vid_padding', type=int, default=75)
    p.add_argument('--txt_padding', type=int, default=200)
    p.add_argument('--num_classes', type=int, default=28)
    p.add_argument('--batch_size',  type=int, default=4)
    p.add_argument('--num_workers', type=int, default=2)

    # Arquitetura Transformer (deve bater com o modelo treinado)
    p.add_argument('--d_model',         type=int, default=512)
    p.add_argument('--nhead',           type=int, default=8)
    p.add_argument('--num_layers',      type=int, default=2)
    p.add_argument('--dim_feedforward', type=int, default=2048)

    p.add_argument('--out_dir', default='results', help='Diretório de saída')
    p.add_argument('--device',  default='auto',    help='cuda | mps | cpu | auto')

    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
