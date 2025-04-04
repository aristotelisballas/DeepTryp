import argparse
import sys

from typing import List, Tuple
from pathlib import Path

from helper_code import find_records
from teamcode.dataset.recording import load_recording

def find_label_distributions(data_folder: Path, dataset:str):
    filenames = find_records(str(data_folder))

    positive = 0
    negative = 0
    positive_filenames: List = []
    negative_filenames: List = []

    processed = 0
    total = len(filenames)
    for filename in filenames:
        recording = load_recording(data_folder, filename)
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
    # with open(f'{dataset}_positive.txt', 'w') as f:
    #     for file in positive_filenames:
    #         f.write(f"{file}\n")
    # f.close()
    #
    # with open(f'{dataset}_negative.txt', 'w') as f:
    #     for file in negative_filenames:
    #         f.write(f"{file}\n")
    # f.close()

    print(f"Positive: {positive}, Negative: {negative}, Total: {len(filenames)}")


# Parse arguments.
def get_parser():
    description = 'Find label distributions for datasets'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-n', '--dataset_name', type=str, required=True)
    return parser

def run(args):
    find_label_distributions(Path(args.data_folder), args.dataset_name)

if __name__ == "__main__":
    run(get_parser().parse_args(sys.argv[1:]))