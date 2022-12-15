from torch.utils.data import Dataset
import pandas as pd
import json
import os
import numpy as np
import random
import torch
from bisect import bisect

from utils.audio import N_FRAMES, log_mel_spectrogram
NUM_CLASSES = 2
classes = ('No Porg-Rock', 'Prog-Rock')


def get_dataset(path):
    df = pd.read_csv(path)
    return df.values.tolist()


class SongDataset(Dataset):
    def __init__(self, annotations, audio_dir='', transform=None, target_transform=None):
        self.audio_labels = annotations
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_labels[idx][0])
        audio = torch.load(audio_path)
        label = self.audio_labels[idx][1]

        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            label = self.target_transform(label)

        start = random.randint(0,audio.shape[1]-(N_FRAMES+1))
        return audio[:,start:N_FRAMES+start], label


class SongDataset_v2(Dataset):
    def __init__(self, annotations, audio_dir='', transform=None, target_transform=None):
        self.audio_labels = annotations
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_labels[idx][0])
        audio = log_mel_spectrogram(audio_path)
        label = self.audio_labels[idx][1]

        if self.transform:
            audio = self.transform(audio)
        if self.target_transform:
            label = self.target_transform(label)

        start = random.randint(0,audio.shape[1]-(N_FRAMES+1))
        return audio[:,start:N_FRAMES+start], label


class SongDatasetTest(Dataset):
    def __init__(self, annotations, audio_dir=''):
        self.audio_labels = annotations
        self.audio_dir = audio_dir

        current_value = 0
        self.song_id = [0]
        
        for idx in range(len(self.audio_labels)):
            audio_path = os.path.join(self.audio_dir, self.audio_labels[idx][0])
            num = torch.load(audio_path).shape[1]//N_FRAMES
            current_value += num
            self.song_id += [current_value]
            
    def __len__(self):
        return self.song_id[-1]

    def __getitem__(self, idx):
        real_index = bisect(self.song_id,idx)-1
        offset = idx - self.song_id[real_index]

        audio_path = os.path.join(self.audio_dir, self.audio_labels[real_index][0])
        audio = torch.load(audio_path)
        
        label = self.audio_labels[real_index][1]

        return audio[:,N_FRAMES*offset:N_FRAMES*(offset+1)], label


class SongDatasetTest_v2(Dataset):
    def __init__(self, annotations, audio_dir=''):
        self.audio_labels = annotations
        self.audio_dir = audio_dir

        current_value = 0
        self.song_id = [0]
        
        for idx in range(len(self.audio_labels)):
            audio_path = os.path.join(self.audio_dir, self.audio_labels[idx][0])
            num = log_mel_spectrogram(audio_path).shape[1]//N_FRAMES
            current_value += num
            self.song_id += [current_value]
            
    def __len__(self):
        return self.song_id[-1]

    def __getitem__(self, idx):
        real_index = bisect(self.song_id,idx)-1
        offset = idx - self.song_id[real_index]

        audio_path = os.path.join(self.audio_dir, self.audio_labels[real_index][0])
        audio = log_mel_spectrogram(audio_path)
        
        label = self.audio_labels[real_index][1]

        return audio[:,N_FRAMES*offset:N_FRAMES*(offset+1)], label
