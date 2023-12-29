import numpy as np
from mindspore.mindrecord import FileWriter
import os
import argparse
import collections
import glob
import gc

cv_schema_json = {"data": {"type": "int32", "shape": [-1]}}

map_dict = {'L': 5, 'S': 6, 'A': 7, 'G': 8, 'E': 9, 'V': 10, 'T': 11, 'R': 12, 'D': 13, 'I': 14,
            'P': 15, 'K': 16, 'N': 17, 'F': 18, 'Q': 19, 'Y': 20, 'H': 21, 'M': 22, 'C': 23, 'W': 24,
            'X': 1, 'B': 2, 'Z': 3, 'U': 4, 'O': 26, 'SOT': 25, 'SHT': 27, 'MED': 28, 'LON': 29, 'EOT': 0, 'PAD': 0}


def encode(seq):
    return [map_dict[c] for c in seq]


def process(data_file):
    data = []
    with open(data_file, 'r') as f:
        for line in f:
            line_split = line.strip().split(" ")
            sample = collections.OrderedDict()
            sample["data"] = np.array(encode(line_split[0]), dtype=np.int32)
            data.append(sample)
    md_name = os.path.join(save_dir, os.path.basename(data_file) + '.mindrecord')
    print(">>>>>>>>>>>>>>>>>save data:", md_name)
    writer = FileWriter(file_name=md_name, shard_num=1)
    writer.add_schema(cv_schema_json, "train_schema")
    writer.write_raw_data(data)
    writer.commit()
    del writer
    del data
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Total data processing")
    parser.add_argument('--data_url',
                        type=str,
                        required=True,
                        default=None,
                        help="data dir")
    parser.add_argument('--save_dir',
                        type=str,
                        default="mindrecord data dir",
                        help="save dir")
    args = parser.parse_args()
    save_dir = args.save_dir
    data_path = args.data_url
    files = glob.glob(data_path + "/*.txt")
    os.makedirs(save_dir, exist_ok=True)
    for file in files:
        process(data_file=file)
