import os
import random

import pandas as pd
import shutil

from pathlib import Path
from typing import List, Tuple

from loguru import logger
from sklearn.model_selection import train_test_split

from teamcode.dataset.recording import load_recording, Recording
import teamcode.challengeconfig as cconf


def get_train_val_test_idx(
        train_perc: float = 0.7, val_perc: float = 0.1, *, num_recordings: int
):
    n1: int = int(train_perc * num_recordings)
    n2: int = int(val_perc * num_recordings)

    train_idx = list(range(0, n1))
    val_idx = list(range(n1, n1 + n2))
    test_idx = list(range(n1 + n2, num_recordings))

    return train_idx, val_idx, test_idx

def select_by_indices(x: List, idx: List[int]) -> List:
    y = list()
    for i in idx:
        y.append(x[i])

    return y

def my_find_records(data_folder:Path) -> List:
    data_folder = str(data_folder)
    records = []
    # Walk through the directory
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.hea'):
                full_path = os.path.abspath(os.path.join(root, file))[:-len('.hea')]
                records.append(full_path)

    return records

def find_exams_csv(data_folder:Path) -> List[pd.DataFrame]:
    data_folder = str(data_folder)
    exams = []
    # Walk through the directory
    for root, _, files in os.walk(data_folder):
        for file in files:
            # print(files)
            if file == 'exams.csv':
                full_path = os.path.abspath(os.path.join(root, file))
                df = pd.read_csv(full_path)
                exams.append(df)

    return exams

def prepare_datasplits(filenames: List, split_type: str = 'mixed', exams: List[pd.DataFrame] = None) \
        -> Tuple[List, List, List[Recording]]:
    logger.info(f'Preparing {split_type} data splits')

    # p_split_file = Path(f'teamcode/dataset/datasplits/{split_type}_positive.txt')
    # n_split_file = Path(f'teamcode/dataset/datasplits/{split_type}_positive.txt')
    #
    # if p_split_file.is_file():
    #     logger.info(f'Found label files, will skip splitting')
    #     with open(p_split_file, 'r') as f:
    #         positive_filenames = [x.strip("\n") for x in f.readlines()]
    #     f.close()
    #
    #     with open(n_split_file, 'r') as f:
    #         negative_filenames = [x.strip("\n") for x in f.readlines()]
    #     f.close()
    #
    # else:
    #     logger.info(f'Did not find label files, will begin splitting')
    all_recordings = []

    positive = 0
    negative = 0
    positive_filenames: List = []
    negative_filenames: List = []

    processed = 0
    total = len(filenames)
    for filename in filenames:
        recording = load_recording(filename, exams)
        all_recordings.append(recording)
        # print(recording.id)
        # print(recording.has_chagas)
        if recording.has_chagas:
            positive += 1
            positive_filenames.append(filename)
        else:
            negative += 1
            negative_filenames.append(filename)
        processed += 1
        print(f"Processed {processed} of {total}", end='\r')

    # Path("teamcode/dataset/datasplits/").mkdir(parents=True, exist_ok=True)

    # with open(f'teamcode/dataset/datasplits/{split_type}_positive.txt', 'w') as f:
    #     for file in positive_filenames:
    #         f.write(f"{file}\n")
    # f.close()
    #
    # with open(f'teamcode/dataset/datasplits/{split_type}_negative.txt', 'w') as f:
    #     for file in negative_filenames:
    #         f.write(f"{file}\n")
    # f.close()
    logger.info(f'Found {len(positive_filenames)} recordings with Chagas and {len(negative_filenames)} without')

    return negative_filenames, positive_filenames, all_recordings


# def split_stratified_scikit(negative_files: List[str], positive_files: List[str]) \
#         -> Tuple[List[str], List[str], List[str]]:
#
#     # Create a combined list of filenames with corresponding labels
#     files = negative_files + positive_files
#     labels = [0] * len(negative_files) + [1] * len(positive_files)  # 0 for negative, 1 for positive
#
#     # First, split into train and remaining (for validation and test)
#     train_files, remaining_files, train_labels, remaining_labels = train_test_split(
#         files, labels, test_size=0.3, stratify=labels, random_state=42
#     )
#
#     # Now split the remaining into validation and test
#     val_files, test_files, val_labels, test_labels = train_test_split(
#         remaining_files, remaining_labels, test_size=0.8, stratify=remaining_labels, random_state=42
#     )
#
#     def calculate_positive_percentage(l):
#         return sum(l) / len(l) * 100
#
#     train_pos_percent = calculate_positive_percentage(train_labels)
#     val_pos_percent = calculate_positive_percentage(val_labels)
#     test_pos_percent = calculate_positive_percentage(test_labels)
#
#     # Output the percentages
#     logger.info(f"Percentage of positive labels in train set: {train_pos_percent:.2f}%")
#     logger.info(f"Percentage of positive labels in validation set: {val_pos_percent:.2f}%")
#     logger.info(f"Percentage of positive labels in test set: {test_pos_percent:.2f}%")
#
#     return train_files, val_files, test_files

def split_stratified(negative_files: List[str], positive_files: List[str],
                     return_splits=False):

    n_train_idx, n_val_idx, n_test_idx = get_train_val_test_idx(num_recordings=len(negative_files))
    p_train_idx, p_val_idx, p_test_idx = get_train_val_test_idx(num_recordings=len(positive_files))

    n_train_files = select_by_indices(negative_files, n_train_idx)
    n_val_files = select_by_indices(negative_files, n_val_idx)
    n_test_files = select_by_indices(negative_files, n_test_idx)

    p_train_files = select_by_indices(positive_files, p_train_idx)
    p_val_files = select_by_indices(positive_files, p_val_idx)
    p_test_files = select_by_indices(positive_files, p_test_idx)

    logger.info(f"Chagas recordings (1): Train - {len(p_train_idx)}, Val - {len(p_val_idx)}, Test - {len(p_test_idx)}")
    logger.info(f"Normal recordings (0): Train - {len(n_train_idx)}, Val - {len(n_val_idx)}, Test - {len(n_test_idx)}")

    train_files = p_train_files + n_train_files
    val_files = p_val_files + n_val_files
    test_files = p_test_files + n_test_files

    # Shuffle the combined data to ensure randomness
    random.seed(42)
    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)

    if return_splits:
        return n_train_files, p_train_files, val_files, test_files
    else:
        return train_files, val_files, test_files



def list_first_level_dirs(directory):
    """Returns a list of first-level directories in the specified directory."""
    try:
        return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    except FileNotFoundError:
        print(f"Error: The directory '{directory}' does not exist.")
        return []
    except PermissionError:
        print(f"Error: Permission denied for accessing '{directory}'.")
        return []


def move_test_files(data_folder, target_dir):
    data_folder: Path = Path(data_folder)
    filenames = my_find_records(data_folder)
    filenames.sort()

    n_files, p_files, _ = prepare_datasplits(filenames, cconf.split_type, None)
    _, _, test_files = split_stratified(n_files, p_files)

    for filename in test_files:
        file = filename.split('/')[-1]
        shutil.copyfile(filename + '.hea', Path(target_dir) / str(file + '.hea'))
        shutil.copyfile(filename + '.dat', Path(target_dir) / str(file + '.dat') )

    print("Copied test files.")