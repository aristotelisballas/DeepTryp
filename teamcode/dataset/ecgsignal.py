import random
from pathlib import Path
from typing import List

import numpy as np
from scipy.signal import firwin, lfilter, resample

from helper_code import load_signals
# from teamcode import challengeconfig as cconf
from teamcode.challengeconfig import wsize_sec, wstep_sec, signal_fs, recording_crop_sec
from teamcode.dataset.recording import Recording
from teamcode.dataset.augmentations import ECGAugmentations


def load_ecg_signal_windows(
        recordings: List[Recording],
        wsize: float = wsize_sec,  # in secs
        wstep: float = wstep_sec,  # in secs
        ) -> [List[np.ndarray], List[np.ndarray]]:
    """
    This functions loads all window sequences from a list of recordings.
    :param recordings:
    :param wsize:
    :param wstep:
    :return:
    """
    ecg_signal_windows: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for recording in recordings:
        recording_signal_windows, recording_labels = load_recording_windows(recording, wsize, wstep)
        ecg_signal_windows.extend(recording_signal_windows)
        labels.extend(recording_labels)

    return ecg_signal_windows, labels


def load_ecg_signal_windows_single(
        recording: Recording,
        wsize: float = wsize_sec,  # in secs
        wstep: float = wstep_sec,  # in secs
        ) -> [List[np.ndarray], List[np.ndarray]]:
    """
    This function loads the whole list of ECG signal windows from a recording
    :param recording:
    :param wsize:
    :param wstep:
    :return:
    """
    ecg_signal_windows: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    recording_signal_windows, recording_labels = load_recording_windows(recording, wsize, wstep)
    ecg_signal_windows.extend(recording_signal_windows)
    labels.extend(recording_labels)

    return ecg_signal_windows, labels

def load_ecg_signal_window_sequence(
        recording: Recording,
        idx,
        wsize: float = wsize_sec,  # in secs
        wstep: float = wstep_sec,  # in secs
        ) -> (np.ndarray, np.ndarray):
    """
    This function loads a specific window from the whole list of ECG signal windows from a recording
    :param recording:
    :param idx:
    :param wsize:
    :param wstep:
    :return:
    """

    ecg_signal_windows: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    recording_signal_windows, recording_labels = load_recording_windows(recording, wsize, wstep)
    ecg_signal_windows.extend(recording_signal_windows)
    labels.extend(recording_labels)

    return ecg_signal_windows[idx], labels[idx]


def load_ecg_signal(
        recording: Recording,
        wsize: float = wsize_sec,  # in secs
        wstep: float = wstep_sec,  # in secs
        signal_only: bool = False,
        augmentor: ECGAugmentations = None,
        cconf: dict = None
        ) -> (np.ndarray, np.ndarray):
    """
    This function loads a single ECG signal from a recording. It is meant to be used when the splitting
    of the recording is not needed. If the specified wsize is larger than the recording length, the signal is
    truncated to that length, whereas if it is smaller it is zero-padded.
    :param cconf:
    :param augmentor:
    :param recording:
    :param idx:
    :param wsize:
    :param wstep:
    :param signal_only:
    :return:
    """

    data = load_recording_windows(recording, wsize, wstep, split=False, signal_only=signal_only,
                                  augmentor=augmentor, cconf=cconf)

    return data


def load_recording_windows(recording: Recording,
                           wsize: float,
                           wstep: float,
                           split: bool = True,
                           signal_only: bool = False,
                           augmentor: ECGAugmentations = None,
                           cconf: dict = None):

        wsize: int = round(wsize * cconf['signal_fs'])
        wstep: int = round(wstep * cconf['signal_fs'])

        recording_signal_windows: List[np.ndarray] = []
        recording_labels: List[int] = []

        rec_label = int(recording.has_chagas)
        rec_len = recording.num_samples
        fs = recording.fs

        # Load ecg windows
        signal, _ = load_signals(str(Path(recording.location) / str(recording.id)))  # array shape is (channels, sig_len)

        # Apply standard pre-processing
        signal = preprocess_signal(signal, fs, cconf)

        # Apply augmentation (randomly)
        if augmentor is not None and random.random() >= 0.5:
            signal = augmentor.augment(signal)

        if split:
            # Crop audio
            signal = crop_sig(signal, rec_len)

            # Extract windows
            ecg_signal_windows = extract_recording_ecg_windows(signal, rec_len, wsize, wstep)

            # Append audio windows and labels
            recording_signal_windows.extend(ecg_signal_windows)
            recording_labels.extend(len(ecg_signal_windows) * [rec_label])

            return recording_signal_windows, recording_labels

        else:
            signal = resize_signal(signal, wsize)

            if signal_only:
                return signal.T
            else:
                return signal.T, rec_label


def extract_recording_ecg_windows(signal: np.ndarray, rec_len:int, wsize: int, wstep: int) -> List[np.ndarray]:
    """
    Creates ecg signal windows from an ecg recording.

    :param signal: The ecg recording signal
    :param rec_len: The length of the recording
    :param wsize: The size of the window (in samples)
    :param wstep: The step of the window (in samples)
    :return: A list of audio windows
    """
    recording_audio_windows: List[np.ndarray] = []

    for i in range(0, rec_len - wsize, wstep):
        recording_audio_windows.append(signal[i:i + wsize, :].T) # Change the dims to (channels, wsize)

    return recording_audio_windows


def normalize_channel(signal: np.ndarray) -> np.ndarray:
    for i in range(signal.shape[0]):
        if np.ptp(signal[i]) == 0:
            signal[i, :] = signal[i, :]
        else:
            signal[i, :] = 2. * (signal[i] - np.min(signal[i])) / (np.ptp(signal[i])) - 1

    return signal


def resample_signal(signal: np.ndarray, output_sampling_rate: int, input_sampling_rate: int) -> np.ndarray:
    if int(output_sampling_rate) == int(input_sampling_rate):
        return signal

    else:
        sample = signal.astype(np.float32)
        factor = output_sampling_rate / input_sampling_rate
        len_old = sample.shape[1]
        num_of_leads = sample.shape[0]
        new_length = int(factor * len_old)
        f_resample = np.zeros((num_of_leads, new_length))

        for i in range(signal.shape[0]):
            f_resample[i, :] = (resample(signal[i], new_length))

        return f_resample


def preprocess_signal(signal: np.ndarray, fs: int, cconf: dict) -> np.ndarray:
    """
    Pre-process audio after loading the signal from the .dat file.
    The input is assumed to be the result of `load_signals` from
    `helper_code.py`, i.e. a 2D numpy array with shape `(signal_length, 12)`.

    First, the values are linearly normalized in [-1.0, 1.0].
    Then, they are resampled to the target fs based on challengeconfig.py
    Finally, they converted to 16-bit floats.

    :param signal: The output of `load_signals` from `helper_code.py`
    :param fs: The sampling frequency of the recording
    :return: The post-processed audio signal
    """

    signal = normalize_channel(signal)

    if cconf['signal_fs'] is not None and cconf['signal_fs'] != fs:
        # signal = resample(audio, int(audio.size * cconf.audio_fs / fs))
        signal = resample_signal(signal, cconf['signal_fs'], fs)

    if cconf['lp_filter']:
        lowpass_kernel = lp_kernel(cconf=cconf)
        signal = lfilter(lowpass_kernel, 1.0, signal)

    return signal


def crop_sig(signal: np.ndarray, signal_len: int) -> np.ndarray:
    n: int = round(recording_crop_sec * signal_fs)

    return signal[n:signal_len - n, :]


def resize_signal(signal: np.ndarray, target_length: int) -> np.ndarray:
    """
    Truncate from the middle or zero-pad a 2D signal array along the time dimension.

    Parameters:
    signal (np.ndarray): Input 2D array of shape (time_steps, channels).
    target_length (int): Desired number of time steps.

    Returns:
    np.ndarray: Resized 2D array of shape (target_length, channels).
    """
    current_length, num_channels = signal.shape
    resized_signal = np.zeros((target_length, num_channels))  # Initialize with zeros

    if current_length > target_length:
        # Truncate from the middle
        start_idx = (current_length - target_length) // 2
        end_idx = start_idx + target_length
        resized_signal = signal[start_idx:end_idx, :]
    else:
        # Zero-pad (center-align the original values)
        start_idx = (target_length - current_length) // 2
        resized_signal[start_idx:start_idx + current_length, :] = signal

    return resized_signal


def lp_kernel(cutoff_freq=20, num_taps=101, cconf: dict=None):
    lowpass_kernel = firwin(num_taps, cutoff=cutoff_freq, fs=cconf['signal_fs'], window="hamming")

    return lowpass_kernel
