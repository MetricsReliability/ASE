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


class IO:
    def load_datasets(self, config):
        data = []
        if config['experiment_mode'] == 1:
            raw_data = pd.read_csv(config['single_dataset'], index_col=None)
            raw_data = DataPreprocessing.remove_useless_attr(raw_data)
            data = raw_data
        else:
            os.chdir(config['multiple_datasets'])
            output_format_extension = 'csv'
            _dataList = [i for i in glob.glob('*.{}'.format(output_format_extension))]
            for i in range(len(_dataList)):
                _path_to_data = config['multiple_datasets'] + _dataList[i]
                _data_i = pd.read_csv(_path_to_data, index_col=None)
                data.append(DataPreprocessing.remove_useless_attr(_data_i))

        return data

    def write_csv(self, data_obj, filename):
        with open(filename, 'w', newline='') as csv_file:
            wr = csv.writer(csv_file)
            for val in data_obj:
                wr.writerow(val)
            csv_file.close()
