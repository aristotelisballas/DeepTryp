import torch
import torch.nn.functional as F
import random
import numpy as np


class ECGAugmentations:
    def __init__(self, noise_std=0.01, shift_max=100, mask_max=200,
                 cutoff_freq=25, fs=500, p_augment=0.5):
        """
        Initialize augmentation parameters.
        - noise_std: Standard deviation for Gaussian noise.
        - shift_max: Max shift (in time steps) for time shift.
        - mask_max: Max number of consecutive time steps to mask.
        - cutoff_freq: Low-pass filter cutoff frequency in Hz.
        - fs: Sampling frequency in Hz.
        """
        self.noise_std = noise_std
        self.shift_max = shift_max
        self.mask_max = mask_max
        self.cutoff_freq = cutoff_freq
        self.fs = fs
        self.p_augment = p_augment
        # self.lowpass_kernel = self._design_lowpass_filter()

    # def _design_lowpass_filter(self, num_taps=101):
    #     """Designs a low-pass FIR filter using a Hamming window."""
    #     kernel = firwin(num_taps, cutoff=self.cutoff_freq, fs=self.fs, window="hamming")
    #     kernel = torch.tensor(kernel, dtype=torch.float32).view(1, 1, -1)  # Shape (1, 1, num_taps)
    #     return kernel
    #
    # def apply_lowpass_filter(self, ecg):
    #     """Applies a low-pass filter to the ECG signal."""
    #     num_taps = self.lowpass_kernel.shape[-1]
    #     padding = num_taps // 2  # Pad to maintain signal length
    #     ecg_filtered = F.conv1d(ecg, self.lowpass_kernel.to(ecg.device), padding=padding, groups=ecg.shape[1])
    #     return ecg_filtered

    def add_gaussian_noise(self, ecg):
        """Adds Gaussian noise to the ECG signal."""
        if random.random() > self.p_augment:
            return ecg
        noise = torch.randn_like(ecg) * self.noise_std
        return ecg + noise

    def time_shift(self, ecg):
        """Shifts the ECG forward or backward in time."""
        if random.random() > self.p_augment:
            return ecg
        shift = random.randint(-self.shift_max, self.shift_max)
        return torch.roll(ecg, shifts=shift, dims=-1)

    def amplitude_scaling(self, ecg, min_scale=0.9, max_scale=1.1):
        """Randomly scales the ECG amplitude."""
        if random.random() > self.p_augment:
            return ecg
        scale = torch.FloatTensor(1).uniform_(min_scale, max_scale).item()
        return ecg * scale

    def time_masking(self, ecg):
        """Masks a random section of the ECG to improve robustness."""
        if random.random() > self.p_augment:
            return ecg
        mask_start = random.randint(0, ecg.shape[-1] - self.mask_max)
        mask_end = mask_start + random.randint(10, self.mask_max)
        ecg[:, :, mask_start:mask_end] = 0
        return ecg

    def lead_permutation(self, ecg):
        """Randomly permutes leads to introduce lead ordering variations."""
        if random.random() > self.p_augment:
            return ecg
        permuted_indices = torch.randperm(ecg.shape[1])  # Shuffle 12 leads
        return ecg[:, permuted_indices, :]

    def augment(self, ecg):
        """Applies a random set of augmentations including the low-pass filter."""
        ecg = self.add_gaussian_noise(ecg)
        ecg = self.time_shift(ecg)
        ecg = self.amplitude_scaling(ecg)
        ecg = self.time_masking(ecg)
        ecg = self.lead_permutation(ecg)
        # ecg = self.apply_lowpass_filter(ecg)  # Apply filtering at the end
        return ecg

# # Example usage
# augmentor = ECGAugmentations()
# sample_ecg = torch.randn(1, 12, 5000)  # Simulating a random ECG tensor
# augmented_ecg = augmentor.augment(sample_ecg)
