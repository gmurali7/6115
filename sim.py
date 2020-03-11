
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import threading

from layers import *
from core import *

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

layers = [[None for layer in range(3)] for core in range(16)] 
for core in range(16):
    layers
    
Conv(input_size=(5,5,3),  filter_size=(3,3,3,32),  stride=1, pad1=1, pad2=1),
Conv(input_size=(5,5,32), filter_size=(3,3,32,32), stride=1, pad1=1, pad2=1),
Conv(input_size=(5,5,32), filter_size=(3,3,32,64), stride=1, pad1=1, pad2=1),
Conv(input_size=(5,5,64), filter_size=(3,3,64,64), stride=1, pad1=1, pad2=1),

model = Model(layers=layers)

####

tests = [
(1, (32, 32), model)
]

####

for test in tests:
    num_example, input_shape, model = test
    x = init_x(num_example, input_shape, 0, 127)
    assert (np.min(x) >= 0 and np.max(x) <= 127)
    y = model.forward(x=x)
    print (np.shape(y))
    
####








