from typing import List

import numpy as np
import itertools
import pandas as pd
import torch
from torch.utils.data import Dataset

from teamcode.dataset.ecgsignal import *
from teamcode.dataset.recording import load_recording
from teamcode.dataset.augmentations import ECGAugmentations

class ECGDataset(Dataset):
    def __init__(self, filenames: List,     # List of lists (first is the negative samples, second is the positive
                 wsize: float = wsize_sec,  # in secs
                 wstep: float = wstep_sec,  # in secs
                 preprocess: bool = True,
                 dataset_path: Path = None,
                 split: bool = None,
                 exams: List[pd.DataFrame] = None,
                 augmentor: ECGAugmentations = None,
                 cconf: dict = None):
        """
        ECG Dataset that returns windows of ECG signals.

        :param recordings: List of filenames objects.
        :param wsize: Window size in seconds.
        :param wstep: Window step in seconds.
        :param preprocess: Whether to preprocess the signals or not.
        """
        self.filenames = filenames
        self.wsize = wsize
        self.wstep = wstep
        self.preprocess = preprocess
        self.dataset_path = dataset_path
        self.nums_sequences: List[int] = list()
        self.recordings: List[Recording] = list()
        self.split: bool = split
        self.exams = exams
        self.augmentor = augmentor
        self.cconf = cconf

        if split:
            for filename in self.filenames:
                # rec_duration_sec: float = len(recording.imu) / signal_fs
                recording = load_recording(dataset_path, filename)
                self.recordings.append(recording)
                if recording.fs != signal_fs:
                    factor = signal_fs / recording.fs
                    new_num_samples = int(factor * recording.num_samples)
                    window_num = len(range(0, new_num_samples - round(wsize_sec * signal_fs),
                                           round(wstep_sec * signal_fs)))
                else:
                    window_num = len(range(0, recording.num_samples - round(wsize_sec * signal_fs),
                                           round(wstep_sec * signal_fs)))

                self.nums_sequences.append(window_num)
            self.nums_sequences_cumsum = np.cumsum(self.nums_sequences)

    def __len__(self):
        """
        Return the total number of signal windows across all recordings.
        """
        if self.split:
            return np.sum(self.nums_sequences)
        else:
            return len(self.filenames)

    def __getitem__(self, idx):
        if self.split:
            recording_idx = np.searchsorted(self.nums_sequences_cumsum, idx, side='right')
            sequence_idx = idx - (self.nums_sequences_cumsum[recording_idx] - self.nums_sequences[recording_idx])

            recording = self.recordings[recording_idx]

            ecg, label = load_ecg_signal_window_sequence(recording, sequence_idx)

            return torch.tensor(ecg, dtype=torch.float32), torch.tensor(label, dtype=torch.float)
        else:
            recording = load_recording(self.filenames[idx], self.exams)
            ecg, label = load_ecg_signal(recording, cconf=self.cconf, augmentor=self.augmentor)
            # print(ecg.shape)

            return torch.tensor(ecg, dtype=torch.float32), torch.tensor(label, dtype=torch.float)



class BalancedECGDataset(Dataset):
    def __init__(self, filenames: List,     # List of lists (first is the negative samples, second is the positive
                 wsize: float = wsize_sec,  # in secs
                 wstep: float = wstep_sec,  # in secs
                 preprocess: bool = True,
                 dataset_path: Path = None,
                 exams: List[pd.DataFrame] = None,
                 augmentor: ECGAugmentations = None,
                 cconf: dict = None):
        """
        ECG Dataset that returns windows of ECG signals.

        :param recordings: List of filenames objects.
        :param wsize: Window size in seconds.
        :param wstep: Window step in seconds.
        :param preprocess: Whether to preprocess the signals or not.
        """
        self.n_files = filenames[0]
        self.p_files = filenames[1]
        self.wsize = wsize
        self.wstep = wstep
        self.preprocess = preprocess
        self.dataset_path = dataset_path
        self.nums_sequences: List[int] = list()
        self.recordings: List[Recording] = list()
        self.exams = exams
        self.augmentor = augmentor
        self.cconf = cconf

        # Get max length to allow infinite sampling
        self.max_len = max(len(self.n_files), len(self.p_files))

        # If one list is shorter, cycle it infinitely
        self.n_cycle = itertools.cycle(self.n_files) if len(self.n_files) < self.max_len else self.n_files
        self.p_cycle = itertools.cycle(self.p_files) if len(self.p_files) < self.max_len else self.p_files


    def __len__(self):
        return self.max_len

    def __getitem__(self, _):
        neg_sample = random.choice(self.n_files)
        pos_sample = random.choice(self.p_files)

        n_rec = load_recording(neg_sample, self.exams)
        p_rec = load_recording(pos_sample, self.exams)

        n_ecg, n_label = load_ecg_signal(n_rec, cconf=self.cconf, augmentor=self.augmentor)
        p_ecg, p_label = load_ecg_signal(p_rec, cconf=self.cconf, augmentor=self.augmentor)

        # n_ecg = torch.tensor(n_ecg, dtype=torch.float32)
        # n_label = torch.tensor(n_label, dtype=torch.float)

        return (torch.tensor(n_ecg, dtype=torch.float32), torch.tensor(n_label, dtype=torch.float)), \
            (torch.tensor(p_ecg, dtype=torch.float32), torch.tensor(p_label, dtype=torch.float))

def balanced_collate_fn(batch):
    """
    Custom collate function for the BalancedECGDataset.
    Ensures each batch has equal positive and negative samples.
    """
    neg_samples, pos_samples = zip(*batch)  # Unzip batch

    # Extract ECG signals and labels separately
    neg_ecgs, neg_labels = zip(*neg_samples)
    pos_ecgs, pos_labels = zip(*pos_samples)

    # Stack ECG signals into a batch tensor
    ecgs = torch.stack(list(neg_ecgs) + list(pos_ecgs))  # (batch_size*2, channels, time)
    labels = torch.cat([torch.stack(neg_labels), torch.stack(pos_labels)])  # Merge labels

    return ecgs, labels


class TripletECGDataset(Dataset):
    def __init__(self, filenames: List,     # List of lists (negatives, positives)
                 wsize: float = wsize_sec,
                 wstep: float = wstep_sec,
                 preprocess: bool = True,
                 dataset_path: Path = None,
                 exams: List[pd.DataFrame] = None,
                 augmentor: ECGAugmentations = None,
                 cconf: dict = None):
        """
        ECG Dataset for pairwise ranking loss (triplet loss setup).
        """
        self.n_files = filenames[0]
        self.p_files = filenames[1]
        self.wsize = wsize
        self.wstep = wstep
        self.preprocess = preprocess
        self.dataset_path = dataset_path
        self.exams = exams
        self.augmentor = augmentor
        self.cconf = cconf

        # Ensure that each batch has both classes balanced
        self.max_len = max(len(self.n_files), len(self.p_files))

    def __len__(self):
        return self.max_len

    def __getitem__(self, _):
        """
        Returns (anchor, positive, negative) samples for pairwise ranking loss.
        """
        anchor_sample = random.choice(self.p_files) # random positive
        pos_sample = random.choice(self.p_files)
        neg_sample = random.choice(self.n_files)

        # Load ECG signals and labels
        anchor_rec = load_recording(anchor_sample, self.exams)
        p_rec = load_recording(pos_sample, self.exams)
        n_rec = load_recording(neg_sample, self.exams)

        a_ecg, _ = load_ecg_signal(anchor_rec, cconf=self.cconf, augmentor=self.augmentor)
        p_ecg, _ = load_ecg_signal(p_rec, cconf=self.cconf, augmentor=self.augmentor)
        n_ecg, _ = load_ecg_signal(n_rec, cconf=self.cconf, augmentor=self.augmentor)

        return (torch.tensor(a_ecg, dtype=torch.float32),
                torch.tensor(p_ecg, dtype=torch.float32),
                torch.tensor(n_ecg, dtype=torch.float32))

def triplet_collate_fn(batch):
    """
    Custom collate function for TripletECGDataset.
    Ensures each batch contains anchor, positive, and negative samples.
    """
    anchors, positives, negatives = zip(*batch)  # Unzip batch into triplets

    # Stack ECG signals into tensors
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)

    return anchors, positives, negatives