import numpy as np
from scipy import stats
import pandas as pd
import libs.entropy_estimators as ee


def selection_sort(pval, index):
    n = len(pval)
    for i in range(n):
        jmin = i
        for j in range(i + 1, n):
            if pval[j] < pval[jmin]:
                jmin = j
        if jmin != i:
            temp1 = pval[i]
            temp2 = index[i]
            pval[i] = pval[jmin]
            index[i] = index[jmin]
            pval[jmin] = temp1
            index[jmin] = temp2
    return pval, index


def feature_selection(tr, ts):
    tr = tr.drop(
        [tr.columns[0]], axis='columns')
    [m, N] = np.shape(tr)
    tr_cmi = np.zeros((N - 1, N - 1))

    single_var_cmi = np.zeros((1, N - 1))
    p_store_tr_single = np.zeros((1, N - 1))

    p_store_tr = np.zeros((N - 1, N - 1))

    u_tr = []
    u_ts = []

    for k in range(0, N - 1):
        unique1, counts1 = np.unique(tr.iloc[:, k], return_counts=True)
        u_tr.append(len(unique1))
    # columns_names = ["loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e", "b", "t", "lOCode", "lOComment",
    #                  "lOBlank", "locCodeAndComment", "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount"]
    #
    # index_names = ["loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e", "b", "t", "lOCode", "lOComment",
    #                  "lOBlank", "locCodeAndComment", "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount"]

    columns_names = ["wmc", "dit", "noc", "cbo", "rfc", "lcom", "ca", "ce", "npm", "lcom3", "loc", "dam", "moa", "mfa",
                     "cam", "ic", "cbm", "amc", "max_cc", "avg_cc"]

    index_names = ["wmc", "dit", "noc", "cbo", "rfc", "lcom", "ca", "ce", "npm", "lcom3", "loc", "dam", "moa", "mfa",
                     "cam", "ic", "cbm", "amc", "max_cc", "avg_cc"]

    tr_cmi = pd.DataFrame(np.zeros((N - 1, N - 1)), columns=columns_names, index=index_names)
    p_store_tr = pd.DataFrame(np.zeros((N - 1, N - 1)), columns=columns_names, index=index_names)

    for i in range(N - 1):
        for j in range(N - 1):
            if i != j:
                tr_cmi.iloc[i, j] = ee.cmidd(tr.iloc[:, i], tr.iloc[:, j], tr.iloc[:, -1])
    tr_cmi = tr_cmi.round(1)
    tr_cmi.to_csv('ant17.csv', index=True, header=True)

    tr_cmi = 2 * m * tr_cmi
    ############################################################
    # for n in range(N - 1):
    #     single_var_cmi[0, n] = ee.midd(tr[:, n], tr[:, -1])
    # single_var_cmi = 2 * m * single_var_cmi
    #
    # for n in range(N - 1):
    #     d1 = 2 * (u_tr[n] - 1)
    #     p_store_tr_single[0][n] = stats.chi2.pdf(single_var_cmi[0, n], d1)
    # single_var_cmi = np.argsort(single_var_cmi)
    # selected_idx = []
    #
    # idx = np.size(single_var_cmi, 1)
    # k = 0
    # while k != 5:
    #     selected_idx.append(single_var_cmi[0, idx - 1])
    #     idx -= 1
    #     k += 1
    ################################################################
    for j in range(N - 1):
        for i in range(N - 1):
            if i != j:
                d1 = 2 * (u_tr[j] - 1) * (u_tr[i] - 1)
                p_store_tr.iloc[j, i] = stats.chi2.cdf(tr_cmi.iloc[j, i], d1)
    #p_store_tr = p_s tore_tr.round(10)
    p_store_tr.to_csv('ant17pvalue.csv', index=True, header=True)
    latent_variables = []
    p_value_vec_tr = []
    index_vec_tr = []
    for i in range(N - 1):
        for j in range(N - 1):
            if i < j:
                p_value_vec_tr.append(p_store_tr[i][j])
                index_vec_tr.append([i, j])
    [p_val_tr, index_tr] = selection_sort(p_value_vec_tr, index_vec_tr)

    k = 0
    for i in range(len(p_val_tr)):
        latent_variables.append(index_tr[i])
        k += 1
        if k >= 10:
            break

    latent_tuples = [tuple(item) for item in latent_variables]
    p1 = []
    p2 = []
    for v1, v2 in latent_tuples:
        p1.append(v1)
    p1 = set(p1)
    new_latent_list = []
    if len(p1) != len(p2):
        for i in range(min(len(p1), len(p2))):
            new_latent_list.append([list(p1)[i], list(p2)[i]])
    p1 = list(p1)
    p1.append(N - 1)
    # p1 is CMI(X,Y|C)
    # selected_idx is CMI(X_i|C)
    selected_idx.append(N - 1)
    tr = tr[:, selected_idx]
    ts = ts[:, selected_idx]
    return tr, ts


def main():
    ant16 = "E:\\apply\\york\\project\\source\\datasets\\file_level\\raw_original_datasets\\ant\\ant-1.7.csv"
    data = pd.read_csv(ant16)
    feature_selection(data, data)


if __name__ == '__main__':
    main()
