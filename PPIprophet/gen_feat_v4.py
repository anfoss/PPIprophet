import re
import sys
import os
import numpy as np
from scipy import stats
import pandas as pd
from scipy.ndimage import uniform_filter
import multiprocessing as mp
from dask import dataframe as dd


import PPIprophet.io_ as io
import PPIprophet.stats_ as st


class ProteinProfile(object):
    """
    ProteinProfile represents a protein's intensity profile and provides methods for peak detection.

    Attributes:
        acc (str): The accession identifier for the protein.
        inten (np.ndarray): The intensity values for the protein profile, converted to a NumPy array of floats.
        peaks (list): List of detected peak indices in the intensity profile.

    Methods:
        __init__(acc, inten):
            Initializes the ProteinProfile with an accession and intensity values.

        get_inte():
            Returns the intensity profile as a NumPy array.

        get_acc():
            Returns the protein accession identifier.

        get_peaks():
            Returns the list of detected peak indices.

        calc_peaks(pick=True):
            Detects peaks in the intensity profile.
            If pick is True, uses st.peak_picking to find peaks.
            If pick is False, uses the index of the maximum intensity as the peak.
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

    def get_peaks(self):
        return self.peaks

    def calc_peaks(self, pick=True):
        # we already splitted so we can use the max
        pks = None
        if pick:
            pks = list(st.peak_picking(self.inten)[0])
            self.peaks = [int(x) for x in pks]
        else:
            self.peaks = [int(np.argmax(self.inten))]
        # avoid breakage due to float


class ComplexProfile(object):
    """
    Represents a protein complex profile, encapsulating its members and various computed features.

    Attributes:
        name (str): Name of the complex.
        members (list): List of protein member objects.
        pks (dict): Dictionary of peak information for each member.
        width (float or None): Width of the complex, typically computed as mean FWHM.
        shifts (float or None): Absolute shift between aligned peaks of members.
        cor (list): List of correlation values between member profiles.
        diff (list): List of differences between member intensity profiles.
        pks_ali (dict): Dictionary of aligned peak positions for each member.
        yhat_probs (Any): Placeholder for predicted probabilities (if used).

    Methods:
        __init__(name):
            Initializes a ComplexProfile with the given name.

        add_member(prot):
            Adds a protein member to the complex.

        get_members():
            Returns a list of accession numbers for all members.

        get_name():
            Returns the name of the complex.

        create_matrix():
            Creates a 2D numpy array of intensity profiles for all members.

        get_peaks():
            Yields formatted rows containing peak information for each member.

        format_ids(deli="#"):
            Returns a string identifier for the complex by concatenating member accessions.

        test():
            Placeholder for correlation-based test between members. Returns True.

        calc_corr(pairs, W=10):
            Calculates vectorized correlation between two member profiles using a sliding window.

        align_peaks(skip=100):
            Aligns peaks of all protein members and updates peak information.

        pairwise():
            Performs pairwise comparison between two members, calculating correlation, difference, and shift.

        calc_shift(ids):
            Calculates the absolute shift between aligned peaks of two members.

        calc_diff(p1, p2):
            Calculates the absolute difference between intensity profiles of two members.

        calc_width():
            Computes the width of the complex (currently set to a constant).

        create_row():
            Aggregates all computed features into a single output row for prediction.
    """

    def __init__(self, name):
        super(ComplexProfile, self).__init__()
        self.name = name
        # members needs to be reformat to have a 2d matrix
        self.members = []
        self.pks = {}
        self.width = None
        self.shifts = None
        self.cor = []
        self.diff = []
        self.pks_ali = {}
        self.yhat_probs = None

    def add_member(self, prot):
        self.members.append(prot)

    def get_members(self):
        return [x.get_acc() for x in self.members]

    def get_name(self):
        return self.name

    def create_matrix(self):
        """
        Creates a 2D NumPy array by extracting integer representations from each member.

        Returns:
            np.ndarray: A 2D array where each row corresponds to the integer representation
            of a member in the `self.members` list, as obtained by calling their `get_inte()` method.

        """
        arr = [x.get_inte() for x in self.members]
        return np.array(arr)

    def get_peaks(self):
        """
        yields one formatted row with pks sel and id
        """
        for k in self.pks.keys():
            yield "{}\t{}\t{}".format(k, self.get_name(), self.pks[k])

    def format_ids(self, deli="#"):
        """
        create a complex identifier by contatenating all the acc
        """
        cmplx_members = self.get_members()
        return deli.join(cmplx_members)

    def test(self):
        # cor = np.corrcoef(
        #                  self.members[0].get_inte(),
        #                  self.members[1].get_inte()
        #                  )
        # # if low correlation no need to compute anything
        # if cor[0][-1] <= 0.0:
        #     # print(self.get_members(), str(cor[0][-1]))
        #     return False
        return True

    def calc_corr(self, pairs, W=10):
        """
        Compute the sliding window Pearson correlation between two input vectors.

        This method calculates the correlation between two vectors (obtained from the `get_inte()` method of the objects in `pairs`)
        using a sliding window of length `W`. The correlation is computed in a vectorized manner for efficiency.

        Parameters
        ----------
        pairs : tuple or list of objects
            A tuple or list containing two objects, each implementing a `get_inte()` method that returns a 1D numpy array.
        W : int, optional
            The window length for the sliding window correlation (default is 10).

        Notes
        -----
        - The method applies a uniform filter to smooth the input vectors before computing the correlation.
        - The correlation is computed for each window position, and the result is stored in `self.cor`.
        - The output is padded with NaNs to reach a fixed length of 72.

        Returns
        -------
        None
            The result is stored in the instance variable `self.cor` as a 1D numpy array.
        """
        tmp = []
        a, b = pairs[0].get_inte(), pairs[1].get_inte()

        # a,b are input arrays; W is window length

        am = uniform_filter(a.astype(float), W)
        bm = uniform_filter(b.astype(float), W)

        amc = am[W // 2 : -W // 2 + 1]
        bmc = bm[W // 2 : -W // 2 + 1]

        da = a[:, None] - amc
        db = b[:, None] - bmc

        # Get sliding mask of valid windows
        m, n = da.shape
        mask1 = np.arange(m)[:, None] >= np.arange(n)
        mask2 = np.arange(m)[:, None] < np.arange(n) + W
        mask = mask1 & mask2
        dam = da * mask
        dbm = db * mask

        ssAs = np.einsum("ij,ij->j", dam, dam)
        ssBs = np.einsum("ij,ij->j", dbm, dbm)
        D = np.einsum("ij,ij->j", dam, dbm)
        # add np.nan to reach 72
        self.cor = np.hstack((D / np.sqrt(ssAs * ssBs), np.zeros(9) + np.nan))

    def align_peaks(self):
        """
        align all protein peaks
        """
        # now we need to create the align file for each protein in this cmplx
        pk = [prot.get_peaks() for prot in self.members]
        # check that both proteins have a peak
        if not all([True if x else False for x in pk]):
            return False
        else:
            ali_pk = alligner(pk)
            # mean only 2 peaks
            prot = [x for x in self.get_members()]
            self.pks_ali = dict(zip(prot, ali_pk))
            for k in self.members:
                if k.get_peaks():
                    row = "#".join(map(str, k.get_peaks()))
                else:
                    row = str(self.pks_ali[k.get_acc()])
                pks = row + "\t" + str(self.pks_ali[k.get_acc()])
                self.pks[k.get_acc()] = pks
            return True

    def pairwise(self):
        """
        performs pairwise comparison
        """
        if len(self.members) == 2:
            self.calc_corr(self.members)
            self.calc_diff(*self.members)
            self.calc_shift([x.get_acc() for x in self.members])
        else:
            print(self.name)
            assert False

    def calc_shift(self, ids):
        self.shifts = abs(self.pks_ali[ids[0]] - self.pks_ali[ids[1]])

    def calc_diff(self, p1, p2):
        self.diff = abs(p1.get_inte() - p2.get_inte())

    def calc_width(self):
        # q = 3
        # width = []
        # for prot in self.members:
        #     peak = int(self.pks_ali[prot.get_acc()])
        #     prot_peak = prot.get_inte()[(peak - q) : (peak + q)]
        #     prot_peak = prot.get_inte()
        #     prot_fwhm = st.fwhm(list(prot_peak))
        #     width.append(prot_fwhm)
        # self.width = np.mean(width)
        self.width = 4

    def create_row(self):
        """
        get all outputs and create a row
        """
        dif_conc = ",".join([str(x) for x in self.diff])
        cor_conc = ",".join([str(x) for x in self.cor])
        row_id = self.get_name()
        members = self.format_ids()
        return pd.Series({
            "ID": row_id,
            "MB": members,
            "COR": cor_conc,
            "SHFT": self.shifts,
            "DIF": dif_conc,
            "W": self.width
        })



def add_top(result, item):
    """Inserts item into list of results"""
    length = len(result)
    index = 0
    # if less than lenght and better diff
    while index < length and result[index][1] < item[1]:
        index += 1
    result.insert(index, item)


def minimize(solution):
    """Returns total difference of solution passed"""
    length = len(solution)
    result = 0
    for index, number1 in enumerate(solution):
        for nr_2_indx in range(index + 1, length):
            result += abs(number1 - solution[nr_2_indx])
    return result


def min_sd(aoa):
    """
    Aligns lists of peaks, pads them to equal length, and selects the set of peaks (one from each list)
    corresponding to the column with the minimum standard deviation.

    Parameters
    ----------
    aoa : list of list
        A list of lists, where each inner list contains peak values (can include None).

    Returns
    -------
    list
        A list containing one peak value from each input list, selected such that the set has the minimum
        standard deviation among all possible aligned columns. If no valid peaks are found, returns None.

    Notes
    -----
    - None values are filtered out from each inner list before alignment.
    - Shorter lists are padded by repeating their last value to match the length of the longest list.
    - If an inner list is empty after filtering, None is appended for that position in the result.
    """
    rf_pk = []
    for v in aoa:
        rf_pk.append([x for x in v if x is not None])
    ln = max([len(x) for x in rf_pk])
    rf_pk2 = [x[:ln] for x in rf_pk]
    for short in rf_pk2:
        while len(short) < ln:
            try:
                short.append(short[-1])
            except IndexError as e:
                break
    pkn = pd.DataFrame(rf_pk2)
    # now all peaks detected are alligned rowise
    # calc standard deviation and take index of min sd
    sd = (pkn.apply(lambda col: np.std(col, ddof=1), axis=0)).tolist()
    try:
        sd = sd.index(min(sd))
    except ValueError as e:
        return None
    indx = []
    # for each protein append index of peak
    # input is protA [peak, peak ,peak]
    # indx out is [protA=> peak, protB => peak, protC => peak]
    # order is same because we append from same array
    for mx_indx in rf_pk2:
        try:
            indx.append(mx_indx[sd])
        # if no peak in mx_indx append none
        except IndexError as e:
            indx.append(None)
    return indx


def shortest_path(aoa, max_trial=100):
    """
    Finds the shortest path through a list of lists (aoa), where each sublist represents possible choices at each step.
    The function attempts to build a path by selecting one element from each sublist, minimizing a cost function at each step.

    Args:
        aoa (list of list): A list of lists, where each inner list contains possible elements to choose from at each step.
        max_trial (int, optional): The maximum number of iterations to attempt before giving up. Defaults to 100.

    Returns:
        list or None: A list representing the shortest path found (one element from each sublist in aoa), or None if no solution is found within max_trial attempts.

    Notes:
        - The function relies on external functions `minimize` (to compute the cost of a path) and `add_top` (to insert new paths into the result list).
        - The function uses a greedy approach, always expanding the current best partial solution.
    """
    elements = len(aoa)
    result = [[[x], 0] for x in aoa[0]]
    trial = 1
    while True:
        if trial == max_trial:
            return None
        trial += 1
        sol = result.pop(0)
        # print(sol)
        # Return the top item if it is complete
        if len(sol[0]) == elements:
            return sol[0]
            # Make new solutions with top item
        for peak in aoa[len(sol[0])]:
            new_pk = [sol[0].copy(), 0]
            new_pk[0].append(peak)
            new_pk[1] = minimize(new_pk[0])
            add_top(result, new_pk)


def alligner(aoa):
    """
    Aligns a list of lists by finding the closest matching points across all sublists.

    Parameters:
        aoa (list of list of int/float): A list containing sublists of numerical values.

    Returns:
        list or None:
            - If any sublist is empty, returns None.
            - If there is at least one common element across all sublists, returns a list where each element is the maximum common value, repeated for the length of aoa.
            - Otherwise, attempts to align using the `shortest_path` function. If successful, returns its result.
            - If `shortest_path` fails, aligns using the `min_sd` function and returns its result.

    Notes:
        - The function assumes the existence of `shortest_path` and `min_sd` helper functions.
        - The alignment strategy prioritizes exact matches, then optimal paths, then minimal standard deviation.
    """
    """Finds closest points of a list of lists"""
    # one of arrays is empty
    for x in aoa:
        if not x:
            return None
    # there is the same nr in all array no need to do anything
    candidate = set.intersection(*map(set, aoa))
    if candidate:
        # returns intersect
        return [max(list(candidate))] * len(aoa)
    else:
        pks = shortest_path(aoa)
        if pks:
            return pks
        else:
            pks = min_sd(aoa)
            return pks


def format_hash(temp):
    """
    get a row hash and create a ComplexProfile object
    """
    inten = temp["FT"].split("#")
    members = temp["MB"].split("#")
    tmp = ComplexProfile(temp["ID"])
    for idx, acc in enumerate(members):
        if acc in tmp.get_members():
            continue
        # peak picking already here
        protein = ProteinProfile(acc, inten[idx].split(","))
        protein.calc_peaks()
        tmp.add_member(protein)
    return tmp


def gen_feat(s):
    """
    receive a single row and generate feature calc
    """
    cmplx = format_hash(s)
    if cmplx.align_peaks():
        cmplx.calc_width()
        cmplx.pairwise()
        return cmplx.create_row()
    else:
        return pd.Series({
            "ID": str(0),
            "MB": str(0),
            "COR": str(0),
            "SHFT": str(0),
            "DIF": str(0),
            "W": str(0)
        })


def process_slice(df):
    return df.apply(gen_feat, axis=1).dropna()


# wrapper
def mp_cmplx(filename):
    """
    map complex into 3 vector => cor vectors
    shift peak
    width peak
    w = point for correlation
    cor(A[idx:(idx + w)], B[idx:(idx+w)])
    width = fwhm(A[idx-q:idx+q])
    so q should be 1/2 of w ?
    """
    things, header = [], []
    temp = {}
    print("calculating features for " + filename)
    df = pd.read_csv(filename, sep="\t")
    sd = dd.from_pandas(df, npartitions=8)

    meta = pd.DataFrame({
        "ID": pd.Series(dtype='str'),
        "MB": pd.Series(dtype='str'),
        "COR": pd.Series(dtype='str'),
        "SHFT": pd.Series(dtype='int'),
        "DIF": pd.Series(dtype='str'),
        "W": pd.Series(dtype='int')
    })

    feats = sd.map_partitions(process_slice, meta=meta).compute()
    return feats


def runner(base):
    """
    generate all features from the mapped complexes file
    base = config[GLOBAL][TEMP]filename
    """
    cmplx_comb = os.path.join(base, "ppi.txt")
    print(os.path.dirname(os.path.realpath(__file__)))
    wr = mp_cmplx(filename=cmplx_comb)
    wr = wr[wr['ID'] != '0']
    wr.to_csv(os.path.join(base, "mp_feat_norm.txt"), sep="\t")
