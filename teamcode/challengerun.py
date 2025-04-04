# import teamcode.challengeconfig as cconf
import torch

from teamcode.dataset.ecgsignal import *
from teamcode.dataset.recording import load_recording
from teamcode.helpers.utils import load_config


def my_challenge_run_model(record, model, verbose):
    cconf = load_config(config_path='./teamcode/config.yaml')

    recording = load_recording(Path(record), None)
    ecg = load_ecg_signal(recording, signal_only=True, cconf=cconf)

    ecg = torch.tensor(ecg, dtype=torch.float32)

    output = model(ecg.unsqueeze(0))
    probability_output = torch.sigmoid(output).cpu().detach().numpy()
    binary_output = (probability_output >= 0.5).astype(int)

    return binary_output.item(), probability_output.item()