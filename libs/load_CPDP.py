import pandas as pd


def load_CPDP_datasets():
    ant16 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\WPDP_datasets\\ant\\ant-1.6.csv"
    ant17 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\WPDP_datasets\\ant\\ant-1.7.csv"

    camel12 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\WPDP_datasets\\camel\\camel-1.2.csv"
    camel14 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\WPDP_datasets\\camel\\camel-1.4.csv"

    ivy14 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\WPDP_datasets\\ivy\\ivy-1.4.csv"
    ivy20 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\WPDP_datasets\\ivy\\ivy-2.0.csv"

    jedit32 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\WPDP_datasets\\jedit\\jedit-3.2.csv"
    jedit40 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\WPDP_datasets\\jedit\\jedit-4.0.csv"

    log4j10 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\WPDP_datasets\\log4j\\log4j-1.0.csv"
    log4j11 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\WPDP_datasets\\log4j\\log4j-1.1.csv"

    lucene20 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\WPDP_datasets\\lucene\\lucene-2.0.csv"
    lucene22 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\WPDP_datasets\\lucene\\lucene-2.2.csv"

    poi25 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\WPDP_datasets\\poi\\poi-2.5.csv"
    poi30 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\WPDP_datasets\\poi\\poi-3.0.csv"

    synapse11 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\WPDP_datasets\\synapse\\synapse-1.1.csv"
    synapse12 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\WPDP_datasets\\synapse\\synapse-1.2.csv"

    xalan24 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\WPDP_datasets\\xalan\\camel-2.4.csv"
    xalan25 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\WPDP_datasets\\xalan\\xalan-2.5.csv"

    xerces12 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\WPDP_datasets\\xerces\\xerces-1.2.csv"
    xerces13 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\WPDP_datasets\\xerces\\xerces-1.3.csv"

    _ds_1 = pd.read_csv(filepath_or_buffer=camel14, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=ant16, index_col=None)
    pair1 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=poi30, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=ant16, index_col=None)
    pair2 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=camel12, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=ant17, index_col=None)
    pair3 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=jedit32, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=ant17, index_col=None)
    pair4 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=ant16, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=camel14, index_col=None)
    pair5 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=xerces12, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=jedit40, index_col=None)
    pair6 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=ivy14, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=jedit40, index_col=None)
    pair7 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=lucene22, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=log4j11, index_col=None)
    pair8 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=xalan25, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=lucene22, index_col=None)
    pair9 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=log4j11, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=lucene22, index_col=None)
    pair10 = [_ds_1, _ds_2]
##
    _ds_1 = pd.read_csv(filepath_or_buffer=lucene22, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=xalan25, index_col=None)
    pair11 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=xerces13, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=xalan25, index_col=None)
    pair12 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=xalan25, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=xerces13, index_col=None)
    pair13 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=ivy20, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=xerces13, index_col=None)
    pair14 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=xerces13, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=ivy20, index_col=None)
    pair15 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=synapse12, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=ivy20, index_col=None)
    pair16 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=ivy14, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=synapse11, index_col=None)
    pair17 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=poi25, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=synapse11, index_col=None)
    pair18 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=ivy20, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=synapse12, index_col=None)
    pair19 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=poi30, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=synapse12, index_col=None)
    pair20 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=synapse12, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=poi30, index_col=None)
    pair21 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=ant16, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=poi30, index_col=None)
    pair22 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=synapse11, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=poi25, index_col=None)
    pair23 = [_ds_1, _ds_2]

    _ds_1 = pd.read_csv(filepath_or_buffer=ant16, index_col=None)
    _ds_2 = pd.read_csv(filepath_or_buffer=poi25, index_col=None)
    pair24 = [_ds_1, _ds_2]

    final_pairs = [pair1, pair2, pair3, pair4, pair5, pair6, pair7, pair8, pair9, pair10, pair11, pair12,
                   pair13,
                   pair14, pair15, pair16, pair17, pair18, pair19, pair20, pair21, pair22, pair23, pair24]
    return final_pairs
