from pathlib import Path
from typing import List, Optional

import pandas as pd

from helper_code import *
from teamcode.dataset import (AGE, SEX, LABEL)


def _extract_property(attributes: List[str], attribute_name: str) -> str:
    for attribute in attributes:
        key_value: List[str] = attribute.split(": ")
        if attribute_name == key_value[0][1:]:
            return key_value[1]

    raise ValueError("Attribute '" + attribute_name + "' not found")

class RecordingMetadata:
    def __init__(self, dataset_path: Optional[Path] = None,
                 id: Optional[int] = None,
                 exams: List[pd.DataFrame] = None,):
        def append_optional_part(p: str) -> Optional[Path]:
            return None if dataset_path is None else dataset_path / (str(id) + p)

        def append_optional_md_part(p: str) -> Optional[str]:
            a = []
            if exams is not None:
                for exam in exams:
                    x = exam.loc[exam['exam_id'] == id]
                    if len(x) > 0 and p in x.columns:
                        a.append(x[p].item())
                    # else:
                    #     a.append(None)
            return None if len(a) == 0 else a[0]

        self.location: str = str(dataset_path)
        self.hea_file: Optional[Path] = append_optional_part(".hea")
        self.dat_file: Optional[Path] = append_optional_part(".dat")

        self.dAVb: Optional[str] = append_optional_md_part("1dAVb")
        self.RBBB: Optional[str] = append_optional_md_part("RBBB")
        self.LBBB: Optional[str] = append_optional_md_part("LBBB")
        self.SB: Optional[str] = append_optional_md_part("SB")
        self.AF: Optional[str] = append_optional_md_part("AF")
        self.ST: Optional[str] = append_optional_md_part("ST")
        self.normal_ecg: Optional[str] = append_optional_md_part("normal_ecg")
        self.death: Optional[str] = append_optional_md_part("death")



class Recording:
    def __init__(self, header: str, dataset_path: Optional[Path] = None, exams: List[pd.DataFrame] = None):

        self.id: int = int(get_record_name(header))

        self.location: str = str(dataset_path)

        self.num_samples: int = int(get_num_samples(header))
        self.fs: int = int(get_sampling_frequency(header))

        self.recording_metadata = RecordingMetadata(dataset_path, self.id, exams)

        def f(attribute: str) -> str:
            return get_variable(header, attribute)[0]

        self.age: str = f(AGE)
        self.sex: str = f(SEX)
        self.has_chagas: bool = f(LABEL) == 'True'


def load_recordings(dataset_path: Path) -> List[Recording]:
    # TODO checks that path exists and all...

    # filenames: List[str] = [filename for filename in glob(str(dataset_path) + "/*.txt")]
    filenames = find_records(str(dataset_path))
    # WARNING: this assumes that filenames contain the ID with prefixed 0. If that's not true, you need to get the
    # integer value of the filename and sort with that
    filenames.sort()

    recordings: List[Recording] = []
    for filename in filenames:
        file = Path(dataset_path) / filename
        # with open(str(Path(dataset_path) / filename), 'r') as file:
        #     data: str = file.read()
        #     if len(data) == 0:
        #         warnings.warn("File '" + filename + "' is empty")
        #         continue
        header = load_header(str(file))
        recordings.append(Recording(header, dataset_path))

    return recordings

def load_recording(filepath: Path, exams: List[pd.DataFrame] = None) -> Recording:
    # TODO checks that path exists and all...

    # filenames: List[str] = [filename for filename in glob(str(dataset_path) + "/*.txt")]
    # filenames = find_records(str(dataset_path))
    # WARNING: this assumes that filenames contain the ID with prefixed 0. If that's not true, you need to get the
    # integer value of the filename and sort with that
    # filenames.sort()

    # recordings: List[Recording] = []
    # file = filepath
    # with open(str(Path(dataset_path) / filename), 'r') as file:
    #     data: str = file.read()
    #     if len(data) == 0:
    #         warnings.warn("File '" + filename + "' is empty")
    #         continue
    header = load_header(str(filepath))
    recording = Recording(header, Path(filepath).parent, exams)

    return recording