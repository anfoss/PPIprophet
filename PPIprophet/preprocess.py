# !/usr/bin/env python3

import os
import itertools
import numpy as np
import pandas as pd

import PPIprophet.io_ as io
import PPIprophet.stats_ as st


# standardize and center methods
def center_arr(hoa, fr_nr="all", smooth=False, stretch=(True, 72)):
    norm = {}
    for k in hoa:
        key = hoa[k]
        # if less than 2 real values
        if len([x for x in key if x > 0]) < 2:
            continue
        key = st.impute_namean(key)
        if fr_nr != "all":
            key = key[0:(fr_nr)]
        if smooth:
            key = st.gauss_filter(key, sigma=1, order=0)
        if stretch[0]:
            # input original length wanted length
            key = st.resample(key, len(key), output_fr=stretch[1])
        key = als(key)
        key = st.resize(key)
        norm[k] = list(key)
    return norm


def als(y, lam=10, p=0.5, niter=100, pl=False, fr=75):
    """
    p for asymmetry and λ for smoothness.
    generally 0.001 ≤ p ≤ 0.1
    10^2 ≤ λ ≤ 10^9
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    import matplotlib.pyplot as plt

    y = np.array(y)
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    if pl:
        ax = plt.subplot(111)
        ax.plot(list(range(0, fr)), y, label="not rescaled")
        ax.plot(list(range(0, fr)), z, label="rescaled")
        plt.legend()
        plt.show()
        plt.close()
    return list(z)


def split_peaks(prot_arr, pr, skp=0):
    """
    split peaks in n samples giving skp fractions of window
    returns
    'right_bases': array([32]), 'left_bases': array([7])
    """
    peaks = list(st.peak_picking(prot_arr, height=0.1, width=3))
    left_bases = peaks[1]["left_bases"]
    right_bases = peaks[1]["right_bases"]
    fr_peak = peaks[0]
    ret = {}
    # if no return value or 1 peak
    if len(fr_peak) < 2:
        ret[pr] = prot_arr
        return ret
    for idx, pk in enumerate(fr_peak):
        nm = "_p_".join([pr, str(idx)])
        clean = fill_zeroes(prot_arr, pk, left_bases[idx], right_bases[idx])
        clean = list(als(np.array(clean), fr=len(clean)))
        ret[nm] = clean
    return ret


def fill_zeroes(prot, pk, left_base, right_base):
    """
    check left and right side of peaks and zero if >
    """
    arr = prot.copy()
    arr[:left_base] = [0 for aa in arr[:left_base]]
    arr[right_base:] = [0 for aa in arr[right_base:]]
    right = zero_sequence(arr[pk: len(arr)])
    left = zero_sequence(arr[:pk][::-1])[::-1]
    return left + right


def zero_sequence(arr):
    idx = 0
    k = True
    while k:
        # if we are at end return array
        if idx == len(arr) - 1:
            return arr
        # if current value smaller than next (i.e increasing)
        elif arr[idx] < arr[(idx + 1)]:
            # slice until there
            tmp = arr[:idx]
            l = [0] * (len(arr) - len(tmp))
            return tmp + l
        idx += 1


def gen_pairs(prot, decoy=True, pow=6, thres=0.2):
    """
    generate all possible pairs between proteins
    remove self dupl i.e between same protein but different apex
    """
    import random
    pairs = list(itertools.combinations(list(prot.keys()), 2))
    ppi = []
    idx = 0
    for p in pairs:
        tmp = np.corrcoef(prot[p[0]], prot[p[1]])[0][-1]
        # wgna style
        if tmp > 0 and tmp**pow > thres:
            l1 = ",".join(map(str, prot[p[0]]))
            l2 = ",".join(map(str, prot[p[1]]))
            row = "#".join([l1, l2])
            nm = "ppi_" + str(idx)
            ppi.append("\t".join([nm, "#".join(p), row]))
            # pick random protein pairs as decoy ppi
            dec = random.choice(pairs)
            l1 = ",".join(map(str, prot[dec[0]]))
            l2 = ",".join(map(str, prot[dec[1]]))
            row = "#".join([l1, l2])
            nm = "DECOY_ppi_" + str(idx)
            ppi.append("\t".join([nm, "#".join([x + '_DECOY' for x in dec]), row]))
            idx += 1
    return ppi


def impute_namean(ls):
    """
    impute 0s in list with value in between if neighbours are values
    assumption is if data is gaussian mean of sequential points is best
    """
    idx = [i for i, j in enumerate(ls) if j == 0]
    for zr in idx:
        if zr == 0 or zr == (len(ls) - 1):
            continue
        elif ls[zr - 1] != 0 and ls[zr + 1] != 0:
            ls[zr] = (ls[zr - 1] + ls[zr + 1]) / 2
        else:
            continue
    return ls


def gen_decoy_ppi(df):
    """
    generate a scrambled dataframe from the original one
    returns a concatenated dataframe
    add 'decoy' to name
    """
    arr = df.values
    idx = np.random.rand(*arr.shape).argsort(axis=1)
    # old = np.sum(arr, axis=1)
    arr = np.take_along_axis(arr, idx, axis=1)
    # assert np.sum(arr, axis=1) == old
    pr = [x + '_decoy' for x in list(df.index)]
    df2 = pd.DataFrame(arr, index=pr)
    return pd.concat([df, df2])


def runner(infile, split=False):
    prot = io.read_txt(infile)
    print("preprocessing " + infile)
    # write it for differential stretch it to assert same length
    prot = center_arr(prot)
    prot2 = {}
    if split:
        for pr in prot:
            pks = split_peaks(prot[pr], pr)
            if pks:
                for k in pks:
                    prot2[k] = pks[k]
    else:
        prot2 = prot
    pr_df = io.create_df(prot2)
    # pr_df = gen_decoy_ppi(pr_df)
    base = io.file2folder(infile, prefix="./tmp/")
    # create tmp folder and subfolder with name
    if not os.path.isdir(base):
        os.makedirs(base)
    # write transf matrix
    dest = os.path.join(base, "transf_matrix.txt")
    pr_df.to_csv(dest, sep="\t", encoding="utf-8", index_label="ID")
    ppi = gen_pairs(prot2, decoy=True)
    nm = os.path.join(base, "ppi.txt")
    io.wrout(ppi, nm, ["ID", "MB", "FT"])
    return True
