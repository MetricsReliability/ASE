import pandas as pd
import numpy as np
import csv
import os, glob


class DataPreprocessing:
    @classmethod
    def remove_useless_attr(cls, raw_data):
        raw_data = pd.DataFrame(raw_data)
        raw_data_numeric_only = raw_data.select_dtypes(include=np.number)
        return raw_data_numeric_only

    @classmethod
    def get_metrics_size(cls, data):
        size_holder = []
        for i in range(np.size(data, 1)):
            size_holder.append(len(np.unique(data[:, i])))
        return size_holder


class IO:
    def load_datasets(self, config):
        if config['granularity'] == 1:
            os.chdir(config['file_level_data_address'])
            address_flag = config['file_level_data_address']
        else:
            os.chdir(config['change_level_data_address'])
            address_flag = config['change_level_data_address']

        data = []
        output_format_extension = 'csv'
        dataset_list = [i for i in glob.glob('*.{}'.format(output_format_extension))]
        for i in range(len(dataset_list)):
            _path_to_data = address_flag + dataset_list[i]
            _data_i = pd.read_csv(_path_to_data, index_col=None)
            data.append(DataPreprocessing.remove_useless_attr(_data_i))

        return data, dataset_list

    def write_csv(self, data_obj, filename):
        with open(filename, 'w', newline='') as csv_file:
            wr = csv.writer(csv_file)
            for val in data_obj:
                wr.writerow(val)
            csv_file.close()
