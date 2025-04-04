import torch
import torch.nn as nn

import teamcode.challengeconfig as cconf
import teamcode.models.biodgmodels as biodgmodels
import teamcode.models.biodgmodels_orig as biodgmodels_orig
from teamcode.models.ecgchagasnet import ResNet1d
from teamcode.models.lebill_cnn import LebillCNN
from teamcode.models.resnet18 import ResNet18
from teamcode.models.resnet1d import Net1D
from teamcode.models.test import Tester


def get_model(eval_phase: bool = False, cconf:dict = None) -> nn.Module:
    if cconf['model'] == 'resnet18':
        if cconf['load_pretrained'] and not eval_phase:
            model = ResNet18()
            state_dict = torch.load('teamcode/models/pretrained_weights/resnet18/model_dict.pth',
                           weights_only=True,
                           map_location=cconf['device'])

            for key in list(state_dict.keys()):
                state_dict[key.replace('network.0.', '')] = state_dict.pop(key)

            model.load_state_dict(state_dict, strict=False)
            if cconf['freeze_layers']:
                model = freeze_layers(model)
        else:
            model = ResNet18()

    elif cconf['model'] == 'lebillcnn':
        model = LebillCNN()

    elif cconf['model'] == 'resnet1d':
        model = Net1D(
            in_channels=12,
            base_filters=64,
            ratio=1.0,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            verbose=False,
            n_classes=1)

    elif cconf['model'] == 'ecgchagasnet':
        """
        Using default setup as in original paper.
        """
        signal_len = int(cconf['wsize_sec'] * cconf['signal_fs'])
        model = ResNet1d(input_dim=(12, signal_len),  # (12, 4000)
                         blocks_dim=[(64, signal_len), (128, 1000), (196, 250), (256, 50), (320, 25)],
                         kernel_size=17,
                         dropout_rate=0.8)

    elif cconf['model'] == 'biodgsresnet':
        model = biodgmodels.BioDGSResNet()

    elif cconf['model'] == 'biodgresnet18':
        if cconf['load_pretrained'] and not eval_phase:
            model = biodgmodels_orig.BioDGResNet18()
            model.load_state_dict(
                torch.load('teamcode/models/pretrained_weights/biodgresnet18/model_dict.pth',
                           weights_only=True,
                           map_location=cconf['device']))

            model.fc = nn.Linear(34624, 1)

            if cconf['freeze_layers']:
                model = freeze_layers(model)
        else:
            model = biodgmodels.BioDGResNet18()

    elif cconf['model'] == 'tester':
        model = Tester(int(cconf['wsize_sec'] * cconf['signal_fs']))

    else:
        model_name = cconf['model']
        raise NotImplementedError(f'Model {model_name} is not implemented.')

    return model


def freeze_layers(model):
    for name, layer in model.named_children():
        if name in ['fc']:
            continue
        else:
            for param in layer.parameters():
                param.requires_grad = False
    return model