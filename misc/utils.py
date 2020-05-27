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

final_label_save_address = "C:\\Users\\Nima\\Desktop\\"

drwang_datasets = "E:\\apply\\york\\project\\dataset\\old_datasets\\jedit"
# class file haye mojud dakhele record haye dataset haye dr wang.
project_address = "E:\\apply\\york\\project\\software\\releases\\main source of projects\\jEdit-3.2\\org\\"
# jayi ke class file hay miran baraye metric extraction.
CKJM_ADDRESS = "E:\\apply\\york\\project\\software\\metric extraction frameworks\\ckjm-1.9\\org\\"
JMT_ADDRESS = "E:\\apply\\york\\project\\software\\metric extraction frameworks\\jmt\\"
UNDERSTAND_ADDRESS = "C:\\mujava\\src\\main\\"


def merg():
    ds_addr = "C:\\Users\\Nima\Desktop\\added\\jmt\\jedit\\jedit-4.0.csv"
    out_addr = "C:\\Users\\Nima\Desktop\\added\\jmt\\jedit\\jedit-4.0.csv"
    _ds_ = pd.read_csv(filepath_or_buffer=ds_addr, index_col=None)
    # _oo_metrics = _ds_.iloc[2, 10]
    [m, n] = _ds_.shape
    for i in range(m):
        _ds_.iloc[i, 0] = _ds_.iloc[i, 1] + "." + _ds_.iloc[i, 0]
    _ds_ = _ds_.drop(['category'], axis=1)
    _ds_.to_csv(out_addr, sep=',', index=None, header=True)


def rm_understand_misc_rows(new_list):
    new_ds = new_list[0]
    new_ds_names = new_list[1]
    # del new_ds['camel'][1]
    for key, value in new_ds.items():
        for i in range(len(value)):
            ds = value[i]
            [m, N] = ds.shape
            ds.apply(pd.to_numeric, errors='corece')
            ds = ds.dropna()
            pd.DataFrame.to_csv(ds, path_or_buf=final_label_save_address + new_ds_names[key][i], sep=',',
                                index_label=None, index=None)


def rm_empty_row(new_list):
    new_ds = new_list[0]
    new_ds_names = new_list[1]
    # del new_ds['camel'][1]
    for key, value in new_ds.items():
        for i in range(len(value)):
            ds = value[i]
            [m, n] = ds.shape
            ds.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
            # while (i <= m):
            #     if pd.isna(ds.iloc[i, -1]):
            #         ds.drop(ds.index[[i]], inplace=True)
            #         i = i - 1
            #     i = i + 1
            pd.DataFrame.to_csv(ds, path_or_buf=final_label_save_address + new_ds_names[key][i], sep=',',
                                index_label=None, index=None)


def copy_status_to_new_dataset(new_list, old_list):
    new_ds = new_list[0]
    new_ds_names = new_list[1]
    for i, item in enumerate(old_list):
        old_ds = old_list[0]
        for key, value in old_ds.items():
            if key in new_ds:
                temp_old = value[i]
                temp_new = new_ds[key][i]
                [m, n] = temp_old.shape
                [m2, n2] = temp_new.shape
                for r in range(m):
                    status = temp_old.iloc[r, -1]
                    name = temp_old.iloc[r, 0]
                    parts_1 = name.split(".")
                    for r2 in range(m2):
                        parts_2 = temp_new.iloc[r2, 0].split(".")
                        if parts_2[-1] == parts_1[-1]:
                            temp_new.iloc[r2, -1] = status
                            print(temp_new.iloc[r2, 0])
                pd.DataFrame.to_csv(temp_new, path_or_buf=final_label_save_address + new_ds_names[key][i], sep=',',
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
        # if _ds_.iloc[record, -1] == 0:
        name = _ds_.iloc[record, 0]
        name = name.replace("org.", "")
        clean_files1.append(name)
        clean_files2.append(name)
        dest.append(name)
        source.append(name)

    for i, _file in enumerate(dest):
        _file = _file.replace(".", "\\")
        _file2 = _file.replace(".", "\\")
        _file = UNDERSTAND_ADDRESS + _file + ".java"
        _file2 = project_address + _file2 + ".java"
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


def find_nonoverlapping(s1, s2):
    addr_ckjm = "C:\\Users\\Nima\\Desktop\\added\\synched\\ckjm\\"
    addr_jmt = "C:\\Users\\Nima\\Desktop\\added\\synched\\jmt\\"
    addr_under = "C:\\Users\\Nima\\Desktop\\added\\synched\\understand\\"
    addr_old = "C:\\Users\\Nima\\Desktop\\added\\synched\\old\\"

    s1_ds = s1[0]
    s1_names = s1[1]
    s2_names = s2[1]
    s2_ds = s2[0]
    for key, value in s1_ds.items():
        to_compare = s2_ds[key]
        for i in range(len(value)):
            sub_ds1 = value[i]
            sub_ds2 = to_compare[i]
            diff1to2 = set(sub_ds1.iloc[:, 0]).difference(sub_ds2.iloc[:, 0])
            diff2to1 = set(sub_ds2.iloc[:, 0]).difference(sub_ds1.iloc[:, 0])

            if diff1to2 is not None:
                for item in diff1to2:
                    for index, row in sub_ds1.iterrows():
                        if item == row[0]:
                            print(item)
                            sub_ds1.drop(index, inplace=True)
            if diff2to1 is not None:
                for item in diff2to1:
                    for index, row in sub_ds2.iterrows():
                        if item == row[0]:
                            print(item)
                            sub_ds2.drop(index, inplace=True)
            if diff1to2 is not None:
                pd.DataFrame.to_csv(sub_ds1, path_or_buf=addr_jmt + s1_names[key][i], sep=',',
                                    index_label=None, index=None)
            if diff2to1 is not None:
                pd.DataFrame.to_csv(sub_ds2, path_or_buf=addr_ckjm + s2_names[key][i], sep=',',
                                    index_label=None, index=None)

    return None


def replace_old_with_new_metrics(old_ds, new_ds):
    return new_ds


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
        ckjm = "E:\\apply\\york\\project\\source\\datasets\\file_level\\CKJM_datasets"
        jmt = "E:\\apply\\york\\project\\source\\datasets\\file_level\\JMT_datasets"
        under = "E:\\apply\\york\\project\\source\\datasets\\file_level\\understand_datasets"
        old = "E:\\apply\\york\\project\\source\\datasets\\file_level\\old_datasets_reduced"
        original = "E:\\apply\\york\\project\\dataset\\old_datasets"
        new_ds_seri_name, new_ds_seri, _ = io_obj.load_datasets(configuration, ckjm,
                                                                drop_unused_columns='misc')

        # rm_empty_row([new_ds_seri, new_ds_seri_name])
        # rm_understand_misc_rows([new_ds_seri, new_ds_seri_name])
        old_ds_seri_name, old_ds_seri, _ = io_obj.load_datasets(configuration, original,
                                                                drop_unused_columns='old')

        replace_old_with_new_metrics([old_ds_seri_name, old_ds_seri], [new_ds_seri_name, new_ds_seri])

        # find_nonoverlapping([new_ds_seri, new_ds_seri_name], [old_ds_seri, old_ds_seri_name])

        # copy_status_to_new_dataset([new_ds_seri, new_ds_seri_name], [old_ds_seri, old_ds_seri_name])
    else:
        merg()


if __name__ == '__main__':
    main()

# for key, value in old_ds_seri.items():
#          print(key)
#          ds1 = value[0]
#          ds2 = value[1]
#          [m1, N1] = ds1.shape
#          [m2, N2] = ds2.shape
#          print((m1 + m2) / 2)
#          ds1 = np.array(ds1)
#          ds2 = np.array(ds2)
#          a = np.where(ds1[:,-1] > 0)
#          a1 = np.where(ds2[:, -1] > 0)
#          b = len(a[0]) / m1
#          b1 = len(a1[0]) / m2
#          print((b + b1) / 2)
