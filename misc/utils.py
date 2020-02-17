import os
import errno
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
import littledarwin
from Levenshtein import _levenshtein

from configuration_files.setup_config import LoadConfig
from data_collection_manipulation.data_handler import IO, DataPreprocessing

io_obj = IO()
dp_obj = DataPreprocessing()

new_dataset_address = "C:\\mujava\\src\\main\\"
old_dataset_address = "E:\\apply\\york\\project\\dataset\\test\\"

real_files_address = "E:\\apply\\york\\project\\software\\releases\\main source of projects\\xalan-2.4\\src\\"
dataset_address_for_file_copy = "E:\\apply\\york\\project\\dataset\\file_level\\xalan\\xalan-2.4.csv"

static_target_address = "C:\\mujava\\src\\main\\"


def copy_status_to_new_dataset(new_list, old_list):
    new_ds = new_list[0]
    new_ds_names = new_list[1]
    for i, item in enumerate(old_list):
        old_ds = old_list[0]
        for key, value in old_ds.items():
            if key in new_ds:
                for i, _ds in enumerate(value):
                    temp_old = value[i]
                    temp_new = new_ds[key][i]
                    [m, n] = temp_old.shape
                    [m2, n2] = temp_new.shape
                    for r in range(m):
                        status = temp_old.iloc[r, -1]
                        name = temp_old.iloc[r, 2]
                        for r2 in range(m2):
                            if temp_new.iloc[r2, 1] == name:
                                temp_new.iloc[r2, -1] = status
                                print(temp_new.iloc[r2, 1])
                    pd.DataFrame.to_csv(temp_new, path_or_buf=new_dataset_address + new_ds_names[key][i], sep=',',
                                        index_label=None, index=None)
    return None


def get_list_of_clean_files(f):
    _ds_ = pd.read_csv(filepath_or_buffer=f, index_col=None)
    [m, n] = _ds_.shape
    clean_files1 = []
    clean_files2 = []
    dest = []
    source = []
    for record in range(m):
        if _ds_.iloc[record, -1] == 0:
            clean_files1.append(_ds_.iloc[record, 2])
            clean_files2.append(_ds_.iloc[record, 2])
        dest.append(_ds_.iloc[record, 2])
        source.append(_ds_.iloc[record, 2])

    # for i, _file in enumerate(clean_files1):
    #     _file = _file.replace('.', "\\")
    #     _file2 = _file.replace('.', "\\")
    #     _file = static_target_address + _file + ".java"
    #     _file2 = real_files_address + _file2 + ".java"
    #     clean_files1[i] = _file
    #     clean_files2[i] = _file2

    for i, _file in enumerate(dest):
        _file = _file.replace(".", "\\")
        _file2 = _file.replace(".", "\\")
        _file = static_target_address + _file + ".java"
        _file2 = real_files_address + _file2 + ".java"
        dest[i] = _file
        source[i] = _file2

    return dest, source


def copy_clean_files_for_mutation(dest_clean_files, base_clean_files):
    for i in range(len(dest_clean_files)):
        if os.path.isfile(base_clean_files[i]):
            if not os.path.exists(os.path.dirname(dest_clean_files[i])):
                try:
                    os.makedirs(os.path.dirname(dest_clean_files[i]))
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise

            with open(dest_clean_files[i], "w") as f:
                p = Path(dest_clean_files[i])
                _path = p.parent
                _path = _path.as_posix()
                if os.path.exists(base_clean_files[i]):
                    shutil.copy2(base_clean_files[i], _path)
        else:
            continue

    return None


def main():
    config_indicator = 1
    ch_obj = LoadConfig(config_indicator)
    configuration = ch_obj.exp_configs

    flag = 2

    if flag == 1:

        # please give your target dataset address as input to this function
        # it returns base_files which are absolute addresses of real files in the project's directory
        # dest_files are absolute addresses where the files from project's directory are going to be copied for
        dest_files, base_files = get_list_of_clean_files(dataset_address_for_file_copy)
        # this function copies files from a specified address to a target address and makes directory if the directory and
        # files are not exist.
        copy_clean_files_for_mutation(dest_files, base_files)
    else:
        new_ds_seri_name, new_ds_seri, _ = io_obj.load_datasets(configuration, new_dataset_address,
                                                                drop_unused_columns=False)
        old_ds_seri_name, old_ds_seri, _ = io_obj.load_datasets(configuration, old_dataset_address,
                                                                drop_unused_columns=False)
        copy_status_to_new_dataset([new_ds_seri, new_ds_seri_name], [old_ds_seri, old_ds_seri_name])


if __name__ == '__main__':
    main()
