import argparse
import datetime
import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from teamcode.dataset.augmentations import ECGAugmentations
from teamcode.dataset.ecgdataset import BalancedECGDataset, ECGDataset, balanced_collate_fn
from teamcode.dataset.utils import find_exams_csv, my_find_records, split_stratified, prepare_datasplits
from teamcode.helpers.utils import load_config
from teamcode.losses.focal import FocalLoss
from teamcode.models.model_utils import get_model


def train_model(model, train_loader, val_loader, log_dir: Path, cconf: dict):
    summary_writer = SummaryWriter(log_dir=str(log_dir))

    # Loss Function
    criterion = FocalLoss(alpha=cconf['alpha'], gamma=cconf['gamma'], reduce=True) if cconf["focal"] else nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cconf['lr'], weight_decay=0.0005)

    num_training_steps = cconf['train_steps']
    # validation_interval = max(1, num_training_steps // 10)
    validation_interval = 500
    best_val_loss = float('inf')
    patience_counter = 0
    loss_divergence_counter = 0
    lr_patience_counter = 0
    min_lr = 1e-6
    lr_factor = 0.5

    step_i = 0
    train_iter = iter(train_loader)
    model.train()
    best_model_state = model.state_dict()
    running_loss = 0.0
    while step_i < num_training_steps:
        start_time = time.time()
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(torch.device(cconf['device'])), y.to(torch.device(cconf['device']))
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, y.float().unsqueeze(1) if y.ndim == 1 else y.float())
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        step_i += 1
        summary_writer.add_scalar('batch_loss', loss.item(), step_i)

        if step_i % validation_interval == 0 or step_i == 1:
            avg_loss = running_loss / validation_interval
            logger.info(f"Step {step_i}/{num_training_steps}, Training Loss: {avg_loss:.4f}")
            model.eval()
            val_loss = 0.0
            total_val_batches = len(val_loader)

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(torch.device(cconf['device'])), y.to(torch.device(cconf['device']))
                    output = model(x)
                    val_loss += criterion(output, y.float().unsqueeze(1) if y.ndim == 1 else y.float()).item()

            avg_val_loss = val_loss / total_val_batches
            logger.info(f"Step {step_i}/{num_training_steps}, Validation Loss: {avg_val_loss:.4f}")
            summary_writer.add_scalar('Loss/val', avg_val_loss, step_i)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], step_i)

            if loss_divergence_counter >= 5:
                logger.info(f"Stopping early due to loss divergence at step {step_i}")
                break

            if avg_val_loss < best_val_loss and step_i > 1:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
                lr_patience_counter = 0
                loss_divergence_counter = 0  # Reset divergence counter as well
            else:
                if step_i > 10000:
                    patience_counter += 1
                    lr_patience_counter += 1
                    logger.info(f"Patience: {patience_counter}/10")

            if lr_patience_counter >= 3 and step_i > 30000:
                current_lr = optimizer.param_groups[0]['lr']
                if current_lr > min_lr:
                    new_lr = max(current_lr * lr_factor, min_lr)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    logger.info(f"Reduced learning rate to {new_lr}")
                lr_patience_counter = 0

            if patience_counter >= 10:
                logger.info(f"Early stopping at step {step_i}")
                break
            running_loss = 0.0
            avg_loss = 0.0

            model.train()

    model.load_state_dict(best_model_state)  # Restore best model state
    summary_writer.close()
    return model


# Function to calculate metrics on the test set
def evaluate_model(model, test_loader, cconf:dict):
    logger.info('Beginning evaluation')
    model.eval()  # Set model to evaluation mode
    criterion = nn.BCEWithLogitsLoss()
    test_loss = 0.0

    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(cconf['device']), y.to(cconf['device'])
            output = model(x)

            # Compute loss
            loss = criterion(output, y.unsqueeze(1))
            test_loss += loss.item() * x.size(0)

            # Convert outputs to probabilities
            probs = torch.sigmoid(output).cpu().numpy()
            preds = (probs >= 0.5).astype(int)  # Convert to binary predictions (0 or 1)

            # Store all outputs and labels for metrics calculation
            all_outputs.extend(preds)
            all_targets.extend(y.cpu().numpy())

    # Convert lists to numpy arrays
    all_outputs = np.array(all_outputs).flatten()
    all_targets = np.array(all_targets).flatten()

    # Compute Metrics
    accuracy = accuracy_score(all_targets, all_outputs)
    precision = precision_score(all_targets, all_outputs, zero_division=0)
    recall = recall_score(all_targets, all_outputs, zero_division=0)
    f1 = f1_score(all_targets, all_outputs)
    conf_matrix = confusion_matrix(all_targets, all_outputs)

    # Log results
    logger.info('######## Test Results #######')
    logger.info(f'Test Loss: {test_loss / len(test_loader.dataset):.4f}')
    logger.info(f'Accuracy: {accuracy:.4f}')
    logger.info(f'Precision: {precision:.4f}')
    logger.info(f'Recall: {recall:.4f}')
    logger.info(f'F1 Score: {f1:.4f}')
    logger.info(f'Confusion Matrix:\n{conf_matrix}')

    return test_loss / len(test_loader.dataset)


def my_challenge_train_model(data_folder, model_folder, verbose,
                             model_v=None,
                             lr_v=None,
                             load_pretrained_v=None,
                             freeze_layers_v=None,
                             lp_filter_v=None,
                             augment_v=None,
                             focal_v=None,
                             alpha_v=None,
                             gamma_v=None,
                             update_config_v=None,
                             ):

    # Load & Fix cconf
    cconf = load_config(config_path='./teamcode/config.yaml')
    if update_config_v:
        atts = {"model": model_v,
                "lr": lr_v,
                "load_pretrained": load_pretrained_v,
                "freeze_layers": freeze_layers_v,
                "lp_filter": lp_filter_v,
                "augment": augment_v,
                "focal": focal_v,
                "alpha": alpha_v,
                "gamma": gamma_v}

        for att, att_v in zip(atts.keys(), atts.values()):
            if att_v is not None:
                cconf[att] = att_v

    cconf['device'] = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    logger.info(f'Config: {cconf}')
    # Some checks
    data_folder: Path = Path(data_folder)
    model_folder: Path = Path(model_folder)
    verbose: int = int(verbose)
    # exam_info: Path = Path(data_folder) / "exams.csv"

    # exams = pd.read_csv(exam_info)
    os.makedirs(model_folder, exist_ok=True)

    log_dir = Path(f"{model_folder}/training_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

    logger.remove()
    logger.add(sys.stderr, level=cconf['log_level'])
    logger.add(f"{log_dir}/log.txt", level=cconf['log_level'])
    device = cconf['device']
    logger.info(f'Model folder: {model_folder}')
    logger.info(f'Using device type: {device}')

    # Get all recording filenames
    filenames = my_find_records(data_folder)
    filenames.sort()

    # recordings: List[Recording] = load_recordings(data_folder)
    num_records = len(filenames)
    logger.info(f'Number of recordings: {num_records}')

    exams = find_exams_csv(data_folder)  # Load exams csv files
    if len(exams) == 0: logger.info('No exams file found!')

    n_files, p_files, _ = prepare_datasplits(filenames, cconf['split_type'], exams)

    # with open('./teamcode/dataset/datasplits/mixed_negative.txt') as f:
    #     n_files = f.read().splitlines()

    # with open('./teamcode/dataset/datasplits/mixed_positive.txt') as f:
    #     p_files = f.read().splitlines()

    random.Random(42).shuffle(n_files)

    n_files = n_files[:15000]

    # with open(r'./teamcode/dataset/datasplits/n_files', 'w') as fp:
    #     for item in n_files:
    #         # write each item on a new line
    #         fp.write("%s\n" % item)
    #
    # fp.close()
    #
    # with open(r'./teamcode/dataset/datasplits/p_files', 'w') as fp:
    #     for item in p_files:
    #         # write each item on a new line
    #         fp.write("%s\n" % item)
    # fp.close()

    # train_idx, val_idx, test_idx = get_train_val_test_idx(num_recordings=num_records)

    # n_train_files, p_train_files, val_files, test_files = split_stratified(n_files, p_files, return_splits=True)
    train_files, val_files, test_files = split_stratified(n_files, p_files, return_splits=True)
    # assert len(train_files) + len(test_files) + len(val_files) == len(filenames)

    # Augmentations
    if cconf['augment']:
        augmentor = ECGAugmentations()
    else:
        augmentor = None

    train_ds = ECGDataset(
        filenames=train_files,
        dataset_path=data_folder,
        exams=exams,
        augmentor=augmentor,
        cconf=cconf,
    )

    val_ds = ECGDataset(
        filenames=val_files,
        dataset_path=data_folder,
        split=cconf['split_windows'],
        exams=exams,
        cconf=cconf
    )

    test_ds = ECGDataset(
        filenames= test_files,
        dataset_path=data_folder,
        split=cconf['split_windows'],
        exams=exams,
        cconf = cconf
    )

    logger.info(f'Dataset length: train = {len(train_ds)}, val_ds = {len(val_ds)}, test_ds = {len(test_ds)}')

    # Create DataLoaders for each split
    batch_size = cconf['batch_size']

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                              #num_workers=8, pin_memory=True, prefetch_factor=4)#, collate_fn=balanced_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
                            # num_workers=8, pin_memory=True, prefetch_factor=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
                             # num_workers=8, pin_memory=True, prefetch_factor=4)

    model = get_model(cconf=cconf)

    logger.info(f"Training model: {cconf['model']}")

    model = model.to(cconf['device'])

    model = train_model(model, train_loader, val_loader, log_dir, cconf)

    evaluate_model(model, test_loader, cconf)

    torch.save(model.state_dict(), log_dir / 'model.pth')
    return

if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="Train and evaluate an ECG model.")
    parser.add_argument("--data_folder", type=str, required=False, help="Path to the dataset.")
    parser.add_argument("--model_folder", type=str, required=False, help="Path to save the model.")
    parser.add_argument("--model", type=str, default=None, help="Model type (e.g., resnet18).")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--load_pretrained", type=str2bool, const=True, nargs='?',
                        default=None, help="Load pretrained weights.")
    parser.add_argument("--freeze_layers", type=str2bool, const=True, nargs='?',
                        default=None, help="Freeze layers during training.")
    parser.add_argument("--lp_filter", type=str2bool, const=True, nargs='?',
                        default=None, help="Apply low-pass filter.")
    parser.add_argument("--augment", type=str2bool, const=True, nargs='?',
                        default=None, help="Apply augmentations.")
    parser.add_argument("--focal", type=str2bool, const=True, nargs='?',
                        default=None, help="Use Focal Loss.")
    parser.add_argument("--alpha", type=float, required=False, default=0.25,
                        help="Alpha parameter in Focal Loss.")
    parser.add_argument("--gamma", type=float, required=False, default=2.0,
                        help="Gamma parameter in Focal Loss.")
    parser.add_argument("--update_config", type=str2bool, const=True, nargs='?',
                        default=False, help="Update config params with args.")
    args = parser.parse_args()

    my_challenge_train_model(data_folder=args.data_folder,
                             model_folder=args.model_folder,
                             verbose=True,
                             model_v=args.model,
                             lr_v=args.lr,
                             load_pretrained_v=args.load_pretrained,
                             freeze_layers_v=args.freeze_layers,
                             lp_filter_v=args.lp_filter,
                             augment_v=args.augment,
                             focal_v=args.focal,
                             alpha_v=args.alpha,
                             gamma_v=args.gamma,
                             update_config_v=args.update_config,
                             )
