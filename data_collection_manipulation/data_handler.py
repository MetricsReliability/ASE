import pandas as pd
import numpy as np
import csv
import os, glob
from pathlib import Path
from collections import OrderedDict


###
class DataPreprocessing:
    @classmethod
    def remove_useless_attr(cls, raw_data):
        raw_data = pd.DataFrame(raw_data)
        raw_data_numeric_only = raw_data.select_dtypes(include=np.number)
        return raw_data_numeric_only

    @classmethod
    def get_metrics_size(cls, data):
        data = np.array(data)
        size_holder = []
        # for i in range(np.size(data, 1)):
        size_holder.append([len(np.unique(data[:, i])) for i in range(np.size(data, 1))])
        return size_holder[0]


class IO:
    def preserve_order(self, input_item):
        seen = set()
        return [x for x in input_item if not (x in seen or seen.add(x))]

    def load_datasets(self, config):
        if config['granularity'] == 1:
            os.chdir(config['file_level_data_address'])
            address_flag = config['file_level_data_address']
            p = Path(address_flag)
        else:
            os.chdir(config['change_level_data_address'])
            address_flag = config['change_level_data_address']

        file_addresses = []
        output_format_extension = ['.csv']
        for file_or_directory in p.iterdir():
            if file_or_directory.is_file() and ''.join(file_or_directory.suffixes).lower() in output_format_extension:
                file_addresses.append(file_or_directory)
            elif file_or_directory.is_dir():
                for x in file_or_directory.iterdir():
                    if ''.join(x.suffix).lower() in output_format_extension:
                        file_addresses.append(x)
            else:
                raise NotADirectoryError

        u_ds_seri = []

        [u_ds_seri.append(v.parts[-2]) for v in file_addresses]
        u_ds_seri = self.preserve_order(u_ds_seri)

        df_list = OrderedDict()

        df_list = {ds_i: [[], []] for ds_i in u_ds_seri}

        for ds_i in u_ds_seri:
            i = 0
            for f in file_addresses:
                if f.parts[-2] == ds_i:
                    _ds_ = pd.read_csv(filepath_or_buffer=f, index_col=None)
                    _ds_ = _ds_.drop([_ds_.columns[0], _ds_.columns[1], _ds_.columns[2]], axis='columns')
                    df_list[ds_i][i] = _ds_
                    i += 1

        _ds_names_ = {ds_i: [[], []] for ds_i in u_ds_seri}
        for ds in _ds_names_.keys():
            i = 0
            for f in file_addresses:
                if ds == f.parts[-2]:
                    _ds_names_[ds][i] = f.name
                    i += 1
        return _ds_names_, df_list

    def write_csv(self, data_obj, filename):
        with open(filename, 'w', newline='') as csv_file:
            wr = csv.writer(csv_file)
            for val in data_obj:
                wr.writerow(val)
            csv_file.close()
