# !/usr/bin/env python3

import sys
import os
import itertools
from functools import reduce
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import networkx as nx


from APprophet import io_ as io
from APprophet import danmf, bigclam, edmot, ego_splitter, nnsed, scd, mnmf, symmnmf


@io.timeit
def runner(outf):
    """
    read folder tmp in directory.
    then loop for each file and create a combined file which contains all files
    creates in the tmp directory
    """
    m_adj = pd.read_csv(os.path.join(outf, 'adj_matrix_combined'))
    print(m_adj)
    adj_arr = np.array(m_adj)
    print(adj_arr)
    G = nx.from_numpy_matrix(adj_arr)
    print('Predicting complexes from network\n')
    # test
    clf = danmf.DANMF(layers=[96, 20], iterations=50)
    clf.fit(G)
    ids_d = dict(zip(range(0, len(ids)), ids))
    out = []
    for k,v in clf.get_memberships().items():
        # this returns a dict of list where list is [cluster nr1, nr2, nr3]
        if ids_d.get(k, False):
            print(v)
            try:
                out.append([k, ids_d[k], ";".join(list(map(str, v)))])
            except TypeError as e:
                out.append([k, ids_d[k], str(v)])

    # communities
    out = pd.DataFrame(out, columns=['IDX', 'Identifier', 'Community'])
    outname = os.path.join(outf, "communities.txt")
    out.to_csv(outname, sep="\t", index=False)
