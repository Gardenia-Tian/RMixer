from __future__ import print_function
import numpy as np

from paddle.io import IterableDataset


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list

    def __iter__(self):
        for file in self.file_list:
            with open(file, "r") as rf:
                for l in rf:
                    l = l.strip().split(",")
                    l = [
                        '0' if i == '' or i.upper() == 'NULL' else i for i in l
                    ]  # handle missing values
                    output_list = []
                    output_list.append(np.array(l).astype('float32'))
                    yield output_list
