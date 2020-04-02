import json
import os


def parse_offsets(offset_file):
    assert os.path.exists(offset_file)
    with open(offset_file, 'r') as f:
        offsets = json.load(f)
    return offsets
