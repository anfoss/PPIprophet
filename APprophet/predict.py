# !/usr/bin/env python3


import os
import pandas as pd
import numpy as np



import APprophet.io_ as io

def runner(base, model="./APprophet/APprophet_dnn.h5"):
    """
    get model file and run prediction
    """
    infile = os.path.join(base, "mp_feat_norm.txt")
    X, memo = io.prepare_feat(infile)
    model = tf.keras.models.load_model(modelname)
    yhat_probs = model.predict(X_test, verbose=0)
    pos = np.array(["Yes" if x == 1 else "No" for x in model.predict_cl(X)])
    out = np.concatenate((memo, prob, pos.reshape(-1, 1)), axis=1)
    header = ["ID", "NEG", "POS", "IS_CMPLX"]
    df = pd.DataFrame(out, columns=header)
    df = df[["ID", "POS", "NEG", "IS_CMPLX"]]
    outfile = os.path.join(base, "dnn.txt")
    df.to_csv(outfile, sep="\t", index=False)
    return True
