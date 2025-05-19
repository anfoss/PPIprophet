# !/usr/bin/env python3

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
import re

from PPIprophet import io_ as io


def mcc(y_true, y_pred):
    y_pred_pos = tf.round(tf.clip_by_value(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = tf.round(tf.clip_by_value(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = tf.reduce_sum(y_pos * y_pred_pos)
    tn = tf.reduce_sum(y_neg * y_pred_neg)

    fp = tf.reduce_sum(y_neg * y_pred_pos)
    fn = tf.reduce_sum(y_pos * y_pred_neg)

    numerator = (tp * tn) - (fp * fn)
    denominator = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return tf.where(tf.math.equal(denominator, 0), 0.0, numerator / denominator)


def plot_probability(df, path):
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white")
    fig.set_size_inches(4, 4)
    pl = sns.displot(df, x="Prob", hue="isdecoy", fill=True, element="step")
    plt.xlabel("Probability", fontsize=9)
    plt.ylabel("Density")
    ax.spines["bottom"].set_color("grey")
    ax.grid(color="w", alpha=0.5)
    ax.tick_params(axis="y", which="major", labelsize=9)
    ax.tick_params(axis="x", which="minor", labelsize=6)
    ax.tick_params(axis="x", which="major", labelsize=6)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()


# @io.timeit
# TODO avoid prepare feat and just have direclty a numpy array
def runner(base, modelname="./PPIprophet/model_ppi_144.keras", chunks=True):
    infile = os.path.join(base, "mp_feat_norm.txt")
    model = tf.keras.models.load_model(
        modelname, custom_objects={"mcc": mcc}, compile=False
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[mcc])
    chunk_size = 30000
    missing = ["nan", "na", "", None, "n", "-"]
    arr = np.array([])
    for chunk in pd.read_csv(infile, sep="\t", na_values=missing, chunksize=chunk_size):
        X, memo = io.prepare_feat(chunk, dropw=["W"])
        yhat_probs = model.predict(X, verbose=0)
        if arr.shape[0] > 1:
            arr = np.vstack((arr, np.column_stack([memo, yhat_probs])))
        else:
            arr = np.column_stack([memo, yhat_probs])
    df = pd.DataFrame(arr, columns=["protS", "Prob"])
    df[["ProtA", "ProtB"]] = df["protS"].str.split("#", expand=True)
    df["isdecoy"] = ["DECOY" if "_DECOY" in x else "TARGET" for x in df["ProtA"]]
    df.drop("protS", inplace=True, axis=1)
    df = df[["ProtA", "ProtB", "Prob", "isdecoy"]]
    # cleanup string to remove _p_0 from middle
    unique_proteins = pd.unique(df[["ProtA", "ProtB"]].values.ravel("K"))
    protein_map = {protein: re.sub("_p_.", "", protein) for protein in unique_proteins}
    df["ProtA"] = df["ProtA"].map(protein_map)
    df["ProtB"] = df["ProtB"].map(protein_map)
    df.to_csv(os.path.join(base, "dnn.txt"), sep="\t", index=False)
    plot_probability(df, os.path.join(base, "prob.pdf"))
