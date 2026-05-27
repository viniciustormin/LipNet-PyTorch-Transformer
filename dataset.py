# encoding: utf-8
import math as _math
import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
from cvtransforms import *
import torch
import glob
import re
import copy
import json
import random
import editdistance


# [v2] CTC beam search decode — puro Python, sem libs externas
def ctc_beam_decode(log_probs: torch.Tensor, beam_width: int = 10, blank: int = 0) -> str:
    """
    CTC beam search decode.
    log_probs: (T, vocab_size) log-softmax probs em CPU.
    blank=0; labels começam em 1 (MyDataset.letters[c-1] para c>=1).
    """
    NEG_INF = float('-inf')

    def _logaddexp(a: float, b: float) -> float:
        if a == NEG_INF:
            return b
        if b == NEG_INF:
            return a
        return max(a, b) + _math.log1p(_math.exp(-abs(a - b)))

    # beams: prefix_tuple → (log_prob_blank, log_prob_nonblank)
    beams: dict = {(): (0.0, NEG_INF)}
    V = log_probs.shape[1]

    for t in range(log_probs.shape[0]):
        lp = log_probs[t]
        new_beams: dict = {}

        for prefix, (pb, pnb) in beams.items():
            p_total = _logaddexp(pb, pnb)

            # estende com blank
            new_pb = p_total + lp[blank].item()
            cur = new_beams.get(prefix, (NEG_INF, NEG_INF))
            new_beams[prefix] = (_logaddexp(cur[0], new_pb), cur[1])

            # estende com cada label não-blank
            for c in range(1, V):
                lp_c = lp[c].item()
                # mesmo label no final: só acumula de caminhos que terminam em blank
                new_pnb = (pb + lp_c) if (prefix and prefix[-1] == c) else (p_total + lp_c)
                new_prefix = prefix + (c,)
                cur = new_beams.get(new_prefix, (NEG_INF, NEG_INF))
                new_beams[new_prefix] = (cur[0], _logaddexp(cur[1], new_pnb))

        # mantém apenas os top beam_width beams
        beams = dict(sorted(
            new_beams.items(),
            key=lambda x: _logaddexp(x[1][0], x[1][1]),
            reverse=True,
        )[:beam_width])

    best = max(beams, key=lambda p: _logaddexp(beams[p][0], beams[p][1]))
    return ''.join(MyDataset.letters[c - 1] for c in best).strip()

    
class MyDataset(Dataset):
    letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    def __init__(self, video_path, anno_path, file_list, vid_pad, txt_pad, phase):
        self.anno_path = anno_path
        self.vid_pad = vid_pad
        self.txt_pad = txt_pad
        self.phase = phase
        
        with open(file_list, 'r') as f:
            self.videos = [os.path.join(video_path, line.strip()) for line in f.readlines()]
            
        self.data = []
        for vid in self.videos:
            items = vid.split(os.path.sep)            
            self.data.append((vid, items[-4], items[-1]))
        
                
    def __getitem__(self, idx):
        (vid, spk, name) = self.data[idx]
        vid = self._load_vid(vid)
        anno = self._load_anno(os.path.join(self.anno_path, spk, 'align', name + '.align'))

        if(self.phase == 'train'):
            vid = HorizontalFlip(vid)
          
        vid = ColorNormalize(vid)                   
        
        vid_len = min(vid.shape[0], self.vid_pad)
        anno_len = min(anno.shape[0], self.txt_pad)
        vid = self._padding(vid, self.vid_pad)
        anno = self._padding(anno, self.txt_pad)
        
        return {'vid': torch.FloatTensor(vid.transpose(3, 0, 1, 2)), 
            'txt': torch.LongTensor(anno),
            'txt_len': anno_len,
            'vid_len': vid_len}
            
    def __len__(self):
        return len(self.data)
        
    def _load_vid(self, p): 
        files = os.listdir(p)
        files = list(filter(lambda file: file.find('.jpg') != -1, files))
        files = sorted(files, key=lambda file: int(os.path.splitext(file)[0]))
        array = [cv2.imread(os.path.join(p, file)) for file in files]
        array = list(filter(lambda im: not im is None, array))
        array = [cv2.resize(im, (128, 64), interpolation=cv2.INTER_LANCZOS4) for im in array]
        array = np.stack(array, axis=0).astype(np.float32)
        return array
    
    def _load_anno(self, name):
        with open(name, 'r') as f:
            lines = [line.strip().split(' ') for line in f.readlines()]
            txt = [line[2] for line in lines]
            txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
        return MyDataset.txt2arr(' '.join(txt).upper(), 1)
    
    def _padding(self, array, length):
        array = [array[_] for _ in range(min(array.shape[0], length))]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)
    
    @staticmethod
    def txt2arr(txt, start):
        arr = []
        for c in list(txt):
            arr.append(MyDataset.letters.index(c) + start)
        return np.array(arr)
        
    @staticmethod
    def arr2txt(arr, start):
        txt = []
        for n in arr:
            if(n >= start):
                txt.append(MyDataset.letters[n - start])     
        return ''.join(txt).strip()
    
    @staticmethod
    def ctc_arr2txt(arr, start):
        pre = -1
        txt = []
        for n in arr:
            if(pre != n and n >= start):                
                if(len(txt) > 0 and txt[-1] == ' ' and MyDataset.letters[n - start] == ' '):
                    pass
                else:
                    txt.append(MyDataset.letters[n - start])                
            pre = n
        return ''.join(txt).strip()
            
    @staticmethod
    def wer(predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return wer
        
    @staticmethod
    def cer(predict, truth):        
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return cer
