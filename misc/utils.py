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

new_dataset_address = "C:\\mujava\\src\\main"
old_file_address = "E:\\apply\\york\\project\\dataset\\file_level\\"
real_files_address = "E:\\apply\\york\\project\\software\\releases\\main source of projects\\camel-1.2\\camel-core\\src\\main\\java\\"
static_target_address = "C:\\mujava\\src\\main\\"


def copy_status_to_new_dataset(new_list, old_list):
    for i, old_ds in enumerate(old_list):
        old_ds = dp_obj.binerize_class(old_ds)
        [m, n] = old_ds.shape
        for record in range(m):
            ds_name = old_ds.iloc[record, 1]
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

    extension = ".csv"
    list_of_new_dataset = io_obj.load_multiple_dataset_from_folder(new_dataset_address, extension)
    list_of_old_dataset = io_obj.load_multiple_dataset_from_folder(old_file_address, extension)
    misc_address = ""
    new_ds_seri, new_ds_seri_name, _ = io_obj.load_datasets(configuration, new_dataset_address)
    old_ds_seri, old_ds_seri_name, _ = io_obj.load_datasets(configuration, configuration['file_level_data_address'])
    copy_status_to_new_dataset([new_ds_seri_name, new_ds_seri],[old_ds_seri_name, old_ds_seri])


# dest_clean_files, base_clean_files = get_list_of_clean_files(clean_file_address)
# copy_clean_files_for_mutation(dest_clean_files, base_clean_files)


if __name__ == '__main__':
    main()
