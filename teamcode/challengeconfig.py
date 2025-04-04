from typing import Optional, List, Union
import torch

# Common
signal_fs: int = 400  # (in Hz)
wsize_sec: float = 10.0
wstep_sec: float = 2.5

# Pre-processing
lp_filter = True

# Augmentations
augment = True

recording_crop_sec: float = 0.0  # Crop start and stop of each audio recording, only during training (in total 2*crop_sec)
split_windows = False

TORCH_DEVICE = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Logger Stuff
log_level= 'INFO'  # Log level: TRACE, DEBUG, INFO (default), SUCCESS, WARNING, ERROR, CRITICAL


# Training Stuff
batch_size = 2
lr = 1e-3
epochs = 2
patience = 5
model = 'biodgresnet18' # resnet18, ecgchagasnet, tester
split_type = 'mixed' # 'ood'

load_pretrained = True
freeze_layers = False

# Testing stuff
