import re
import sys
import os
import numpy as np
from scipy import stats
import pandas as pd
from scipy.ndimage import uniform_filter
import multiprocessing as mp


import PPIprophet.io_ as io
import PPIprophet.stats_ as st


class ProteinProfile(object):
    """
    docstring for ProteinProfile
    """

    def __init__(self, acc, inten):
        super(ProteinProfile, self).__init__()
        self.acc = acc
        self.inten = np.array([float(x) for x in inten])
        self.peaks = []

    def get_inte(self):
        return self.inten

    def get_acc(self):
        return self.acc


def format_hash(temp):
    """
    get a row hash and create a ComplexProfile object
    """
    inten = temp["FT"].split("#")
    members = temp["MB"].split("#")
    toret = []
    for idx, acc in enumerate(members):
        toret.append(ProteinProfile(acc, inten[idx].split(",")))
    return toret


def create_row(conv, prod, diff, id, mb):
    """
    get all outputs and create a row
    """
    conv = ",".join([str(x) for x in conv])
    prod = ",".join([str(x) for x in prod])
    diff = ",".join([str(x) for x in prod])
    row = [
        id,
        mb,
        conv,
        prod,
        diff,
        '-1',
        '-1',
    ]
    return "\t".join([str(x) for x in row])


# wrapper
def mp_cmplx(filename, feat_nm, peak_nm):
    """
    map ppi into 3d array with [height, width, color]
    Notice that the first dimension is the height, and the second dimension is the width. That is because the data is ordered by lines, then each line is ordered by pixels, and finally each pixel contains 3 byte values for RGB. Each colour is represented by an unsigned byte (numpy type uint8).
    ppi are formatted as [10, 72, color]

    """
    things, header = [], []
    temp = {}
    feat_file = []
    peaks_file = []
    i = 0
    print("calculating features for " + filename)
    for line in open(filename, "r"):
        line = line.rstrip("\n")
        if line.startswith("ID" + "\t"):
            header = re.split(r"\t+", line)
        else:
            things = re.split(r"\t+", line)
            temp = {}
            temp = dict(zip(header, things))
        if temp:
            prots = format_hash(temp)
            conv = np.convolve(prots[0].get_inte(), prots[1].get_inte())
            prod = prots[0].get_inte() * prots[1].get_inte()
            diff = np.abs(np.subtract(prots[0].get_inte(), prots[1].get_inte()))
            print(conv.shape, prod.shape, diff.shape)
            assert False
            feat_row = create_row(conv, prod, diff, temp['ID'], temp['MB'])
            feat_file.append(feat_row)
            for k in temp['MB'].split('#'):
                row = "{}\t{}\t{}\t{}".format(k, temp['ID'],'1','1')
                peaks_file.append(row)
            # avoid memory error
            if len(feat_file) == 20000:
                io.dump_file(feat_nm, "\n".join(feat_file))
                io.dump_file(peak_nm, "\n".join(peaks_file))
                feat_file = []
                peak_file = []
                i += 1
                print('dump {}'.format(i))
    return feat_file, peaks_file


def runner(base):
    """
    generate all features from the mapped complexes file
    base = config[GLOBAL][TEMP]filename
    """
    cmplx_comb = os.path.join(base, "ppi.txt")
    print(os.path.dirname(os.path.realpath(__file__)))
    feat_header = [
        "ID",
        "MB",
        "COR",
        "SHFT",
        "DIF",
        "W",
    ]
    feat_nm = os.path.join(base, "mp_feat_norm.txt")
    peak_nm = os.path.join(base, "peak_list.txt")
    io.create_file(feat_nm, feat_header)
    io.create_file(peak_nm,  ["MB", "ID", "PKS", "SEL"])
    wr, pks = mp_cmplx(cmplx_comb, feat_nm, peak_nm)
