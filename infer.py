"""
infer.py — Inferência LipNet-Transformer para o GRID Corpus.

O script aceita dois formatos de entrada:
  - Diretório de frames JPEG pré-extraídos (padrão do GRID Corpus)
  - Arquivo de vídeo bruto (.mpg / .mp4), extraindo frames com OpenCV

Uso (diretório de JPEGs — recomendado quando o dataset já está pré-processado):
    python infer.py \\
        --video      data/grid/lip/s1/video/mpg_6000/bgan6p \\
        --checkpoint checkpoints/all/transformer_v2/transformer_best.pt

Uso (vídeo bruto, sem face detection):
    python infer.py \\
        --video      data/grid/raw_videos/s1/video/mpg_6000/bgan6p.mpg \\
        --checkpoint checkpoints/all/transformer_v2/transformer_best.pt
"""

import argparse
import os
import re
import time

import cv2
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from dataset import MyDataset
from models import LipNetTransformer


# ---------------------------------------------------------------------------
# Pré-processamento de vídeo
# ---------------------------------------------------------------------------

def load_from_dir(path: str, vid_pad: int = 75):
    """
    Carrega frames de um diretório de JPEGs numerados (1.jpg, 2.jpg, ...).
    Pipeline idêntico ao MyDataset._load_vid: lê, redimensiona para 128×64,
    empilha em float32.
    Retorna (lista_de_frames_BGR, tensor_pronto_para_modelo).
    """
    files = sorted(
        [f for f in os.listdir(path) if f.lower().endswith('.jpg')],
        key=lambda f: int(os.path.splitext(f)[0]),
    )
    frames = []
    for fname in files:
        img = cv2.imread(os.path.join(path, fname))
        if img is None:
            continue
        # Redimensiona para largura=128, altura=64 (igual ao dataset)
        img = cv2.resize(img, (128, 64), interpolation=cv2.INTER_LANCZOS4)
        frames.append(img)

    return frames, _frames_to_tensor(frames, vid_pad)


def load_from_video(path: str, vid_pad: int = 75):
    """
    Extrai frames de um arquivo de vídeo com OpenCV.
    Redimensiona cada frame para 128×64 sem detecção de lábios
    (para uso com vídeos já recortados ou para demonstração rápida).
    Retorna (lista_de_frames_BGR, tensor_pronto_para_modelo).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (128, 64), interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame)
    cap.release()

    return frames, _frames_to_tensor(frames, vid_pad)


def _frames_to_tensor(frames: list, vid_pad: int) -> torch.Tensor:
    """
    Converte lista de frames BGR (H=64, W=128, C=3) em tensor de modelo.

    Pipeline (idêntico ao MyDataset.__getitem__):
      1. Empilha em array (T, H, W, C) float32
      2. Normaliza: divide por 255.0  ← ColorNormalize do cvtransforms.py
      3. Trunca ou zero-pads para vid_pad frames
      4. Transpõe (T, H, W, C) → (C, T, H, W)
      5. Adiciona dimensão de batch → (1, C, T, H, W)
    """
    arr = np.stack(frames, axis=0).astype(np.float32)  # (T, 64, 128, 3)
    arr = arr / 255.0                                   # normalização: [0, 1]

    T = arr.shape[0]
    if T < vid_pad:
        # Zero-padding temporal até vid_pad
        pad = np.zeros((vid_pad - T, *arr.shape[1:]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    else:
        arr = arr[:vid_pad]

    # (T, H, W, C) → (C, T, H, W) → (1, C, T, H, W)
    tensor = torch.from_numpy(arr.transpose(3, 0, 1, 2)).unsqueeze(0)
    return tensor  # shape: (1, 3, 75, 64, 128)


# ---------------------------------------------------------------------------
# Carregamento do modelo
# ---------------------------------------------------------------------------

def load_model(
    ckpt_path: str,
    device: torch.device,
    num_layers: int = 4,
    d_model: int = 512,
    nhead: int = 8,
    dim_feedforward: int = 2048,
    num_classes: int = 28,
) -> torch.nn.Module:
    """
    Instancia LipNetTransformer, carrega pesos e coloca em modo eval.
    Remove o prefixo '_orig_mod.' gerado pelo torch.compile caso presente.
    """
    model = LipNetTransformer(
        num_layers=num_layers,
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_classes=num_classes,
    )

    state = torch.load(ckpt_path, map_location='cpu')

    # torch.compile salva pesos com prefixo '_orig_mod.' — removemos aqui
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}

    model.load_state_dict(state)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Decodificação CTC
# ---------------------------------------------------------------------------

def ctc_greedy_decode(logits: torch.Tensor) -> str:
    """
    Decodificação greedy CTC: argmax por timestep seguido de colapso CTC.
    Reutiliza MyDataset.ctc_arr2txt (já implementado no projeto).

    logits: (T, num_classes) — saída bruta do modelo para um único exemplo
    Vocabulário: blank=0, classe 1→' ', classe 2→'A', ..., classe 27→'Z'
    """
    indices = logits.argmax(dim=-1).cpu().numpy()  # (T,) — índice mais provável
    return MyDataset.ctc_arr2txt(indices, start=1)  # remove repetidos e blanks


# ---------------------------------------------------------------------------
# Visualização: GIF com texto sobreposto
# ---------------------------------------------------------------------------

def save_gif(frames: list, text: str, out_path: str, fps: int = 25):
    """
    Gera um GIF animado com os frames do vídeo e o texto predito sobreposto.

    frames  : lista de arrays BGR (H=64, W=128)
    text    : transcrição predita
    out_path: caminho de saída (.gif)
    fps     : quadros por segundo (padrão 25, igual ao GRID Corpus)
    """
    pil_frames = []
    for frame in frames:
        # Converte BGR (OpenCV) → RGB (PIL)
        rgb = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Amplia 3× para melhor visualização (64×128 → 192×384)
        img = img.resize((384, 192), Image.NEAREST)

        # Sobrepõe texto predito em amarelo, com sombra preta para contraste
        draw = ImageDraw.Draw(img)
        x, y = 6, 6
        draw.text((x + 1, y + 1), text, fill=(0, 0, 0))    # sombra
        draw.text((x, y),         text, fill=(255, 230, 0)) # texto amarelo

        pil_frames.append(np.array(img))

    imageio.mimsave(out_path, pil_frames, fps=fps, loop=0)
    print(f'GIF salvo: {out_path}')


# ---------------------------------------------------------------------------
# Ground truth a partir de arquivo .align
# ---------------------------------------------------------------------------

def _guess_align_path(video_path: str) -> str:
    """
    Tenta inferir o caminho do arquivo .align a partir do caminho do vídeo.
    Espera o padrão: .../lip/{spk}/video/mpg_6000/{name}[.mpg]
    Retorna o caminho provável ou string vazia se não encontrado.
    """
    # Normaliza separadores e remove extensão se for arquivo
    path = os.path.normpath(video_path)
    name = os.path.splitext(os.path.basename(path))[0]

    # Extrai speaker (padrão: s seguido de dígitos)
    parts = path.split(os.sep)
    spk = next((p for p in parts if re.fullmatch(r's\d+', p)), None)
    if not spk:
        return ''

    # Procura a raiz do projeto (diretório que contém data/)
    candidate = path
    for _ in range(8):
        candidate = os.path.dirname(candidate)
        align_path = os.path.join(
            candidate, 'data', 'grid', 'GRID_align_txt',
            spk, 'align', f'{name}.align',
        )
        if os.path.exists(align_path):
            return align_path

    return ''


def load_ground_truth(align_path: str) -> str:
    """
    Lê um arquivo .align do GRID Corpus e retorna a frase transcrita.
    Descarta tokens de silêncio ('SIL', 'SP') — igual ao MyDataset._load_anno.
    """
    if not align_path or not os.path.exists(align_path):
        return ''
    words = []
    with open(align_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and parts[2].upper() not in ('SIL', 'SP'):
                words.append(parts[2].upper())
    return ' '.join(words)


# ---------------------------------------------------------------------------
# CLI principal
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Inferência LipNet-Transformer no GRID Corpus')

    # Entrada e checkpoint
    p.add_argument('--video',      required=True,
                   help='Diretório de JPEGs pré-extraídos ou arquivo de vídeo (.mpg/.mp4)')
    p.add_argument('--checkpoint', required=True,
                   help='Caminho do checkpoint .pt do modelo')

    # Arquivo de anotação (opcional — auto-detectado se possível)
    p.add_argument('--align', default='',
                   help='Arquivo .align para exibir o ground truth (opcional)')

    # Parâmetros do modelo (devem bater com o checkpoint)
    p.add_argument('--num_layers',      type=int, default=4)
    p.add_argument('--d_model',         type=int, default=512)
    p.add_argument('--nhead',           type=int, default=8)
    p.add_argument('--dim_feedforward', type=int, default=2048)
    p.add_argument('--num_classes',     type=int, default=28)
    p.add_argument('--vid_pad',         type=int, default=75)

    # Saída
    p.add_argument('--gif',    default='demo_output.gif',
                   help='Caminho do GIF de saída')
    p.add_argument('--device', default='auto',
                   help='Dispositivo: auto (MPS→CPU), mps, cpu')

    return p.parse_args()


def main():
    args = parse_args()

    # --- 1. Dispositivo ---
    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Device: {device}')

    # --- 2. Pré-processamento ---
    print(f'Vídeo : {args.video}')
    if os.path.isdir(args.video):
        frames, tensor = load_from_dir(args.video, args.vid_pad)
    elif os.path.isfile(args.video):
        frames, tensor = load_from_video(args.video, args.vid_pad)
    else:
        raise FileNotFoundError(f'Vídeo não encontrado: {args.video}')

    tensor = tensor.to(device)
    print(f'       {len(frames)} frames → tensor {tuple(tensor.shape)}')

    # --- 3. Modelo ---
    model = load_model(
        args.checkpoint, device,
        num_layers=args.num_layers,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        num_classes=args.num_classes,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Modelo: {n_params:,} parâmetros carregados de {args.checkpoint}')

    # --- 4. Inferência ---
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)  # (1, T, num_classes)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # --- 5. Decodificação greedy: argmax por timestep + colapso CTC ---
    pred = ctc_greedy_decode(logits[0])

    # --- 6. Ground truth (opcional) ---
    align_path = args.align or _guess_align_path(args.video)
    gt = load_ground_truth(align_path)

    # --- 7. Saída no terminal ---
    print(f'\n{"="*50}')
    print(f'  Pred  : {pred or "(vazio)"}')
    if gt:
        print(f'  GT    : {gt}')
    print(f'  Tempo : {elapsed_ms:.1f} ms')
    print(f'{"="*50}')

    # --- 8. GIF de demonstração ---
    if frames:
        # Usa apenas os frames reais (sem o padding de zeros)
        vis_frames = frames[:min(len(frames), args.vid_pad)]
        save_gif(vis_frames, pred, args.gif)


if __name__ == '__main__':
    main()
