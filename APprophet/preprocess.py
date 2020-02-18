# !/usr/bin/env python3

import sys
import os
import re

import numpy as np

import APprophet.io_ as io
import APprophet.stats_ as st


# standardize and center methods
def center_arr(hoa, fr_nr="all", smooth=True, stretch=(True, 72)):
    norm = {}
    for k in hoa:
        key = hoa[k]
        # key = baseline_als(key)
        if fr_nr != "all":
            key = key[0:(fr_nr)]
        # if less than 2 real values
        if len([x for x in key if x > 0]) < 2:
            continue
        if smooth:
            key = st.gauss_filter(key, sigma=1, order=0)
        key = st.impute_namean(key)
        if stretch[0]:
            # input original length wanted length
            key = st.resample(key, len(key), output_fr=stretch[1])
        key = st.resize(key)
        norm[k] = list(key)
    return norm


def baseline_als(y, lam=10, p=0.1, niter=10):
    """
    perform baseline correction
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    p for asymmetry and λ for smoothness.
    generally 0.001 ≤ p ≤ 0.1
    10^2 ≤ λ ≤ 10^9
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(list(range(0,75)), y, label='not rescaled')
    y = np.array(y)
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    ax.plot(list(range(0,75)), z, label='rescaled')
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
    peaks = list(st.peak_picking(prot_arr))
    left_bases = peaks[1]["left_bases"]
    right_bases = peaks[1]["right_bases"]
    fr_peak = peaks[0]
    ret = {}
    # if no return value or 1 peak
    if len(fr_peak) < 2:
        ret[pr] = prot_arr
        return ret
    for idx, pk in enumerate(fr_peak):
        if pk < 6 and pk > 69:
            continue
        nm = "_".join([pr, str(idx)])
        clean = fill_zeroes(prot_arr, pk, left_bases[idx], right_bases[idx])
        ret[nm] = clean
    return ret


def fill_zeroes(prot, pk, left_base, right_base):
    """
    check left and right side of peaks and zero if >
    """
    arr = prot.copy()
    arr[:left_base] = [0 for aa in arr[:left_base]]
    arr[right_base:] = [0 for aa in arr[right_base:]]
    right = zero_sequence(arr[pk : len(arr)])
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

def gen_pairs(prot):
    """
    generate all possible pairs between proteins
    remove self dupl i.e between same protein but different apex
    """
    nms = list(prot.keys())
    count = 0
    index = 1
    pairs = []
    for element1 in nms:
        for element2 in nms[index:]:
            if element1[:-1] in element2:
                continue
            else:
                pairs.append((element1, element2))
        index += 1
    ppi = []
    idx = 0
    for p in pairs:
        l1 = ','.join(map(str, prot[p[0]]))
        l2 = ','.join(map(str, prot[p[1]]))
        row = '#'.join([l1, l2])
        nm = 'ppi_' + str(idx)
        ppi.append('\t'.join([nm, '#'.join(p), row]))
        idx +=1
    return ppi


@io.timeit
def runner(infile):
    prot = io.read_txt(infile)
    print("preprocessing " + infile)
    # write it for differential stretch it to assert same length
    prot = center_arr(prot)
    prot2 = {}
    for pr in prot:
        pks = split_peaks(prot[pr], pr)
        if pks:
            for k in pks:
                prot2[k] = pks[k]
    pr_df = io.create_df(prot2)
    base = io.file2folder(infile, prefix="./tmp/")
    # create tmp folder and subfolder with name
    if not os.path.isdir(base):
        os.makedirs(base)
    # write transf matrix
    dest = os.path.join(base, "transf_matrix.txt")
    pr_df.to_csv(dest, sep="\t", encoding="utf-8")
    ppi = gen_pairs(prot2)
    nm = os.path.join(base, "ppi.txt")
    io.wrout(ppi, nm, ["ID", "MB", "FT"])
    return True
