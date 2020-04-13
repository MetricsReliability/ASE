import os
import errno

import pandas
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

drwang_datasets_equivalent_binary_files = "E:\\apply\\york\\project\\software\\releases\\binary releases\\jakarta-log4j-1.0.4\\org\\"

drwang_datasets = "E:\\apply\\york\\project\\source\\datasets\\file_level\\old_datasets"
drwang_datasets_equivalent_source_codes = "E:\\apply\\york\\project\\software\\releases\\main source of projects\\xerces-1.2\\src\\org\\"

new_dataset_address = "C:\\Users\\Nima\\Desktop\\understand\\labaled"
static_target_address = "C:\\mujava\\src\\main\\"

final_label_save_address = "C:\\Users\\Nima\\Desktop\\aaa\\"


def merg():
    ds_addr = ""
    out_addr = ""
    _ds_ = pd.read_csv(filepath_or_buffer=ds_addr, index_col=None)
    _oo_metrics = _ds_.iloc[2, 10]
    [m, n] = _ds_.shape
    for i in range(m):
        _ds_.iloc[i, 0] = _ds_.iloc[i, 0] + "." + _ds_.iloc[i, 1]
    _ds_ = _ds_.drop(['filename'], axis=1)
    _ds_.to_csv(out_addr, sep=',', index=None, header=True)


def copy_status_to_new_dataset(new_list, old_list):
    new_ds = new_list[0]
    new_ds_names = new_list[1]
    for key, value in new_ds.items():
        ds = value[0]
        [m, n] = ds.shape
        i = 0
        ds.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        # while(i <= m):
        #     if pd.isna(ds.iloc[i, -1]):
        #         ds.drop(ds.index[[i]], inplace=True)
        #         i = i - 1
        #     i = i + 1
        pd.DataFrame.to_csv(ds, path_or_buf=final_label_save_address + new_ds_names[key][0], sep=',',
                            index_label=None, index=None)

    #
    # for i, item in enumerate(old_list):
    #     old_ds = old_list[0]
    #     for key, value in old_ds.items():
    #         if key in new_ds:
    #             temp_old = value[0]
    #             temp_new = new_ds[key][0]
    #             [m, n] = temp_old.shape
    #             [m2, n2] = temp_new.shape
    #             for r in range(m):
    #                 status = temp_old.iloc[r, -1]
    #                 name = temp_old.iloc[r, 0]
    #                 parts_1 = name.split(".")
    #                 for r2 in range(m2):
    #                     parts_2 = temp_new.iloc[r2, 0].split(".")
    #                     if parts_2[-1] == parts_1[-1]:
    #                         temp_new.iloc[r2, -1] = status
    #                         print(temp_new.iloc[r2, 0])
    #             pd.DataFrame.to_csv(temp_new, path_or_buf=final_label_save_address + new_ds_names[key][0], sep=',',
    #                                 index_label=None, index=None)
    return None


def get_list_of_clean_files(f):
    _ds_ = pd.read_csv(filepath_or_buffer=f, index_col=None)
    [m, n] = _ds_.shape
    clean_files1 = []
    clean_files2 = []
    dest = []
    source = []
    for record in range(m):
        # if _ds_.iloc[record, -1] == 0:
        name = _ds_.iloc[record, 2]
        name = name.replace("org.", "")
        clean_files1.append(name)
        clean_files2.append(name)
        dest.append(name)
        source.append(name)

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
        _file2 = drwang_datasets_equivalent_source_codes + _file2 + ".java"
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

    flag = 3

    if flag == 1:
        # please give your target dataset address as input to this function
        # it returns base_files which are absolute addresses of real files in the project's directory
        # dest_files are absolute addresses where the files from project's directory are going to be copied for
        dest_files, base_files = get_list_of_clean_files(drwang_datasets)
        # this function copies files from a specified address to a target address and makes directory if the directory and
        # files are not exist.
        copy_clean_files_for_mutation(dest_files, base_files)
    elif flag == 3:
        new_ds_seri_name, new_ds_seri, _ = io_obj.load_datasets(configuration, new_dataset_address,
                                                                drop_unused_columns=False, flag_delete=False)
        old_ds_seri_name, old_ds_seri, _ = io_obj.load_datasets(configuration, drwang_datasets,
                                                                drop_unused_columns=False, flag_delete=True)
        copy_status_to_new_dataset([new_ds_seri, new_ds_seri_name], [old_ds_seri, old_ds_seri_name])
    else:
        merg()


if __name__ == '__main__':
    main()
