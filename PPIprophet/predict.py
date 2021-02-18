# !/usr/bin/env python3

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
import joblib

from PPIprophet import io_ as io



def mcc(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denominator + K.epsilon())

#
def runner(base, modelname="./PPIprophet/APprophet_dnn_no_width.h5"):
    infile = os.path.join(base, "mp_feat_norm.txt")
    model = tf.keras.models.load_model(modelname, custom_objects={'mcc':mcc})
    X, memo = io.prepare_feat(infile, dropw=['W'])
    yhat_probs = model.predict(X, verbose=0)
    df = pd.DataFrame(np.column_stack([memo, yhat_probs]), columns=["protS", "Prob"])
    df["ProtA"], df["ProtB"] = df["protS"].str.split("#", 1).str
    isdecoy = ['DECOY' if '_DECOY' in x else 'TARGET' for x in df['ProtA']]
    df['isdecoy'] = isdecoy
    pred_path = os.path.join(base, "dnn.txt")
    df.drop("protS", inplace=True, axis=1)
    df = df[["ProtA", "ProtB", "Prob", 'isdecoy']]
    print(df[df['Prob']>0.5].shape[0], df.shape[0])
    df.to_csv(pred_path, sep="\t", index=False)
