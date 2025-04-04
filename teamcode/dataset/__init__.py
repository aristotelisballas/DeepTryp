import socket
from pathlib import Path
from typing import List

def root_path() -> Path:
    hostname: str = socket.gethostname()

    if hostname in ['ragnos', 'impact-cluster', 'compute-1', 'compute-2']:
        return Path.home() / 'workspace' / 'datasets' / 'vidimu'
    elif hostname in ['aballas']:
        return Path.home() / 'datasets' / 'physionet2025'
    else:
        return NotImplemented('Unknown hostname')

def path():
    return root_path() / 'code15_wfdb_subset'

# Header Attributes
PATIENT_ID = '# Patient ID:'
ENCOUNTER_ID = '# Encounter ID:'
AGE = '# Age:'
SEX = '# Sex:'
LABEL = '# Chagas label:'
LABEL_PROBA = '# Chagas probability:'

