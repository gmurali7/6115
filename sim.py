
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import threading

from layers import *

####

def init_x(num_example, input_shape, xlow, xhigh):
    h, w = input_shape
    (_, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test[0:num_example, 0:h, 0:w, :]
    
    scale = (np.max(x_test) - np.min(x_test)) / (xhigh - xlow)
    x_test = x_test / scale
    x_test = np.floor(x_test)
    x_test = np.clip(x_test, xlow, xhigh)
    
    x_test = x_test.astype(int)
    return x_test

####

params = {
'bpa': 8,
'bpw': 8,
'adc': 8,
'wl': 128,
'bl': 128,
'offset': 128,
}

weights = np.load('cifar10_weights.npy', allow_pickle=True).item()

####

layers = [
Conv(input_size=(32,32,3),  filter_size=(3,3,3,64),  stride=1, pad1=1, pad2=1, weights=None, params=params),
#Conv(input_size=(32,32,64), filter_size=(3,3,64,64), stride=2, pad1=1, pad2=1, weights=None, params=params),

#Conv(input_size=(16,16,64), filter_size=(3,3,64,128), stride=1, pad1=1, pad2=1),
#Conv(input_size=(16,16,128), filter_size=(3,3,128,128), stride=2, pad1=1, pad2=1),
]

model = Model(layers=layers)
network = model.cut(num_cores=16)

####

tests = [
(1, (32, 32), network)
]

####

for test in tests:
    num_example, input_shape, network = test
    x = init_x(num_example, input_shape, 0, 127)
    assert (np.min(x) >= 0 and np.max(x) <= 127)
    y = network.forward(x=x)
    y_ref = model.forward(x=x)
    print (np.shape(y), np.shape(y_ref))
    assert (np.all(y[0] == y_ref[0]))
    
####








