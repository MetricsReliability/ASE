from collections import OrderedDict
import pandas as pd
import numpy as np
import csv
import os, glob
from pathlib import Path


def rearrange(pivot, ds):
    l2 = OrderedDict()
    for i, v in enumerate(pivot):
        l2[v] = ds[pivot[i]]
    return l2


class DataPreprocessing:
    @staticmethod
    def binerize_class(data):
        for key, value in data.items():
            for i, item in enumerate(value):
                num_records = item.shape[0]
                for r in range(num_records):
                    if item.iloc[r, -1] > 0:
                        item.iloc[r, -1] = 2
                    else:
                        item.iloc[r, -1] = 1
                data[key][i] = item
        return data

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

    def load_multiple_dataset_from_folder(self, path, extension):
        all_files = glob.glob(path + "/*" + extension)
        list_of_files = []
        for filename in all_files:
            df_i = pd.read_csv(filename, index_col=None)
            list_of_files.append(df_i)
        return list_of_files

    def load_datasets(self, config, misc_address=None, drop_unused_columns=True, drop_unused_selection=3):
        if config['granularity'] == 1:
            os.chdir(config['file_level_data_address'])
            address_flag = config['file_level_data_address']
            p = Path(address_flag)
        if config['granularity'] == 2:
            os.chdir(config['change_level_data_address'])
            address_flag = config['change_level_data_address']
        if config['granularity'] == 3:
            p = Path(misc_address)
            os.chdir(misc_address)

        file_addresses = []
        output_format_extension = ['.csv']
        for file_or_directory in p.iterdir():
            if file_or_directory.is_file() and ''.join(file_or_directory.suffix).lower() in output_format_extension:
                file_addresses.append(file_or_directory)
            elif file_or_directory.is_dir():
                for x in file_or_directory.iterdir():
                    if ''.join(x.suffix).lower() in output_format_extension:
                        file_addresses.append(x)
            else:
                continue

        u_ds_seri = []

        # parts[-2] extracts dataset category e.g camel or ant
        # parts[-1] extracts dataset name itself e.g camel-1.0 or ant-1.4
        [u_ds_seri.append(v.parts[-2]) for v in file_addresses]
        u_ds_seri = self.preserve_order(u_ds_seri)

        df_datasets_ = {ds_i: [[], []] for ds_i in u_ds_seri}
        _df_dataset_names_ = {ds_i: [[], []] for ds_i in u_ds_seri}
        _df_file_names = {ds_i: [[], []] for ds_i in u_ds_seri}

        for ds_i in u_ds_seri:
            i = 0
            for f in file_addresses:
                if f.parts[-2] == ds_i:
                    _ds_ = pd.read_csv(filepath_or_buffer=f, index_col=None)
                    _df_file_names[ds_i][i] = _ds_.iloc[:, 0]

                    if drop_unused_columns == 'old':
                        _ds_ = _ds_.drop(
                            [_ds_.columns[0], _ds_.columns[1]], axis='columns')
                        # _ds_ = _ds_.drop(
                        #     [_ds_.columns[0]], axis='columns')


                    df_datasets_[ds_i][i] = _ds_
                    _df_dataset_names_[ds_i][i] = f.name
                    i += 1
        df_datasets_ = rearrange(u_ds_seri, df_datasets_)
        return _df_dataset_names_, df_datasets_, _df_file_names

    def write_csv(self, data_obj, filename):
        with open(filename, 'w', newline='') as csv_file:
            wr = csv.writer(csv_file)
            for val in data_obj:
                wr.writerow(val)
            csv_file.close()
