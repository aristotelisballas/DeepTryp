from pathlib import Path

import torch

# import teamcode.challengeconfig as cconf
from teamcode.dataset.utils import list_first_level_dirs
from teamcode.helpers.utils import load_config
from teamcode.models.model_utils import get_model


def my_challenge_load_model(model_folder, verbose):

    model_folder: Path = Path(model_folder)
    verbose: int = int(verbose)
    # print(model_folder)
    training_dirs = list_first_level_dirs(str(model_folder))
    training_dirs.sort()
    # print(training_dirs)
    ######## Only for Grid Search experimentation
    cconf = load_config(config_path='./teamcode/config.yaml')
    # if 'pretrainedTrue' in training_dirs[-1]:
    #     cconf['load_pretrained'] = True
    # else:
    #     cconf['load_pretrained'] = False
    # if 'biodgresnet18' in training_dirs[-1]:
    #     cconf['model'] = 'biodgresnet18'
    # elif 'biodgsresnet' in training_dirs[-1]:
    #     cconf['model'] = 'biodgsresnet'
    # elif 'lebillcnn' in training_dirs[-1]:
    #     cconf['model'] = 'lebillcnn'
    # else:
    #     cconf['model'] = 'resnet18'

    cconf['device'] = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = get_model(eval_phase=True, cconf=cconf)

    model.load_state_dict(
        torch.load(Path(model_folder) / training_dirs[-1] / 'model.pth',
                   weights_only=True,
                   map_location=cconf['device']))

    return model