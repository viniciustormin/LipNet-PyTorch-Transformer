"""
Unified training script for LipNet-GRU and LipNet-Transformer.

Train GRU model:
    python train_transformer.py --model gru --save_dir checkpoints/gru

Train Transformer model:
    python train_transformer.py --model transformer --save_dir checkpoints/transformer

Both models are trained with identical data, hyperparameters, and seed so
results are directly comparable.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MyDataset
from models import LipNetGRU, LipNetTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(args) -> nn.Module:
    if args.model == 'gru':
        return LipNetGRU(dropout_p=args.dropout, num_classes=args.num_classes)
    return LipNetTransformer(
        dropout_p=args.dropout,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        attn_dropout=args.attn_dropout,
        num_classes=args.num_classes,
    )


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )


def ctc_decode(y: torch.Tensor):
    """Greedy CTC decode. y: (B, T, C)."""
    y = y.argmax(-1)
    return [MyDataset.ctc_arr2txt(y[i], start=1) for i in range(y.size(0))]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(net: nn.Module, args) -> tuple[float, float, float]:
    net.eval()
    dataset = MyDataset(
        args.video_path, args.anno_path, args.val_list,
        args.vid_padding, args.txt_padding, 'test',
    )
    loader = make_loader(dataset, args.batch_size, args.num_workers, shuffle=False)
    crit = nn.CTCLoss(zero_infinity=True)

    losses, wer_list, cer_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            vid = batch['vid'].cuda()
            txt = batch['txt'].cuda()
            vid_len = batch['vid_len'].cuda()
            txt_len = batch['txt_len'].cuda()

            y = net(vid)
            loss = crit(
                y.transpose(0, 1).log_softmax(-1),
                txt, vid_len.view(-1), txt_len.view(-1),
            )
            losses.append(loss.item())

            pred = ctc_decode(y)
            truth = [MyDataset.arr2txt(txt[i], start=1) for i in range(txt.size(0))]
            wer_list.extend(MyDataset.wer(pred, truth))
            cer_list.extend(MyDataset.cer(pred, truth))

    net.train()
    return float(np.mean(losses)), float(np.mean(wer_list)), float(np.mean(cer_list))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    model = build_model(args).cuda()

    if args.weights:
        ckpt = torch.load(args.weights, map_location='cuda')
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f'Loaded weights from {args.weights}')
        if missing:
            print(f'  Missing keys : {missing}')
        if unexpected:
            print(f'  Unexpected   : {unexpected}')

    net = nn.DataParallel(model).cuda()

    train_dataset = MyDataset(
        args.video_path, args.anno_path, args.train_list,
        args.vid_padding, args.txt_padding, 'train',
    )
    loader = make_loader(train_dataset, args.batch_size, args.num_workers, shuffle=True)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=0.0, amsgrad=True,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True,
    )
    crit = nn.CTCLoss(zero_infinity=True)

    history = {
        'model': args.model,
        'train_loss': [], 'val_loss': [],
        'val_wer': [], 'val_cer': [],
        'epochs': [],
    }
    best_wer = float('inf')
    history_path = os.path.join(args.save_dir, 'history.json')

    print(f'\n{"="*60}')
    print(f'  Model : {args.model.upper()}')
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Params: {n_params:,}')
    print(f'  Train : {len(train_dataset)} samples')
    print(f'{"="*60}\n')

    for epoch in range(args.max_epoch):
        net.train()
        epoch_losses = []
        t0 = time.time()

        for i, batch in enumerate(loader):
            vid = batch['vid'].cuda(non_blocking=True)
            txt = batch['txt'].cuda(non_blocking=True)
            vid_len = batch['vid_len'].cuda(non_blocking=True)
            txt_len = batch['txt_len'].cuda(non_blocking=True)

            optimizer.zero_grad()
            y = net(vid)
            loss = crit(
                y.transpose(0, 1).log_softmax(-1),
                txt, vid_len.view(-1), txt_len.view(-1),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(loss.item())

            if (i + 1) % args.display == 0:
                pred = ctc_decode(y)
                truth = [MyDataset.arr2txt(txt[k], start=1) for k in range(txt.size(0))]
                print(f'  [{epoch+1}/{args.max_epoch}] iter {i+1}/{len(loader)} '
                      f'loss={loss.item():.4f}')
                for p, t in list(zip(pred, truth))[:2]:
                    print(f'    pred : {p}')
                    print(f'    truth: {t}')

        train_loss = float(np.mean(epoch_losses))
        val_loss, val_wer, val_cer = validate(net, args)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(f'Epoch {epoch+1:3d}/{args.max_epoch} | '
              f'train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | '
              f'val_wer={val_wer:.4f} | val_cer={val_cer:.4f} | '
              f't={elapsed:.1f}s')

        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_wer'].append(val_wer)
        history['val_cer'].append(val_cer)

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        # Best checkpoint (by WER)
        if val_wer < best_wer:
            best_wer = val_wer
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, f'{args.model}_best.pt'),
            )
            print(f'  -> New best WER: {best_wer:.4f}  (checkpoint saved)')

        # Latest checkpoint (for resuming)
        torch.save(
            model.state_dict(),
            os.path.join(args.save_dir, f'{args.model}_last.pt'),
        )

    print(f'\nTraining complete. Best val WER = {best_wer:.4f}')
    print(f'Checkpoints and history saved to: {args.save_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='LipNet training — GRU or Transformer')

    # Model choice
    p.add_argument('--model', choices=['gru', 'transformer'], default='transformer',
                   help='Sequence model to use')

    # Dataset paths
    p.add_argument('--video_path', default='/data/grid/lip',
                   help='Root dir containing speaker/video-name/ frame folders')
    p.add_argument('--anno_path', default='/data/grid/GRID_align_txt',
                   help='Root dir containing speaker/align/*.align files')
    p.add_argument('--train_list', default='data/overlap_train.txt')
    p.add_argument('--val_list', default='data/overlap_val.txt')

    # Padding (must match dataset structure)
    p.add_argument('--vid_padding', type=int, default=75)
    p.add_argument('--txt_padding', type=int, default=200)
    p.add_argument('--num_classes', type=int, default=28,
                   help='CTC output size: 27 chars + 1 blank')

    # Training hyperparameters (identical for both models)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--max_epoch', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dropout', type=float, default=0.5,
                   help='Spatial dropout applied in STConv frontend')
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--display', type=int, default=50,
                   help='Print sample predictions every N iterations')

    # Transformer-specific hyperparameters
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--dim_feedforward', type=int, default=2048)
    p.add_argument('--attn_dropout', type=float, default=0.1,
                   help='Dropout inside Transformer attention layers')

    # Misc
    p.add_argument('--save_dir', default='checkpoints/transformer',
                   help='Directory to save checkpoints and history.json')
    p.add_argument('--weights', default='',
                   help='Optional pretrained weights to fine-tune from')
    p.add_argument('--gpu', default='0')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    train(args)
