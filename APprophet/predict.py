# !/usr/bin/env python3

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json

from APprophet import io_ as io


def runner(base, modelname="./APprophet/APprophet_dnn.h5"):
    infile = os.path.join(base, "mp_feat_norm.txt")
    model = tf.keras.models.load_model(modelname)
    X, memo = io.prepare_feat(infile)
    yhat_probs = model.predict(X, verbose=0)
    df = pd.DataFrame(np.column_stack([memo, yhat_probs]), columns=["protS", "Prob"])
    df["ProtA"], df["ProtB"] = df["protS"].str.split("#", 1).str
    pred_path = os.path.join(base, "dnn.txt")
    df.drop("protS", inplace=True, axis=1)
    df = df[["ProtA", "ProtB", "Prob"]]
    df.to_csv(pred_path, sep="\t", index=False)
