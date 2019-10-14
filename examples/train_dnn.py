#import setGPU
import keras.backend as K
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import KFold

def layer(din, n_units, do_dropout=True):
    d = Dense(n_units)(din)
    d = LeakyReLU(alpha=0.2)(d)
    if do_dropout:
        d = Dropout(0.2)(d)
    return d


datasets = [
    ("data/TTJets_SemiLeptMGDecays_0_arrs.npy", 0),
    ("data/TTJets_SemiLeptMGDecays_1_arrs.npy", 0),
    ("data/TTJets_SemiLeptMGDecays_2_arrs.npy", 0),
    ("data/GluGluToHToMM_0_arrs.npy", 1),
]

Xs = []
ys = []
for fn, is_signal in datasets:
    data = np.load(fn)
    Xs += [data]
    ys += [is_signal * np.ones((data.shape[0], 1), dtype=np.float32)]

X = np.vstack(Xs)
y = np.vstack(ys)

shuf = np.random.permutation(len(X))
X = X[shuf]
y = y[shuf]


weights = np.ones(y.shape[0], dtype=np.float32)

for cls in np.unique(y):
    sel = (y[:, 0]==cls)
    weights[sel] = 1.0 / np.sum(sel) 

nfeatures = X.shape[1]
ntrain = int(0.8*len(X))

kf = KFold(n_splits=5)
ikf = 0
for train_index, test_index in kf.split(X):
    inp = Input(shape=(nfeatures, ))
    d = layer(inp, 256)
    d = layer(d, 256)
    d = layer(d, 256)
    d = layer(d, 256)
    out = Dense(1, activation="sigmoid")(d)
    model = Model(inp, out)
    optimizer = Adam(0.001)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    model.fit(X[train_index], y[train_index], sample_weight=weights[train_index],
        validation_data=[X[test_index], y[test_index], weights[test_index]],
        epochs=100, batch_size=5000)
    model.save("data/model_kf{0}.h5".format(ikf))
    ikf += 1
