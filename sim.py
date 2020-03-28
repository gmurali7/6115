
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
'rpa': 128
}

weights = np.load('cifar10_weights.npy', allow_pickle=True).item()

####

layers = [
Conv(input_size=(32,32,3),  filter_size=(3,3,3,64),  stride=1, pad1=1, pad2=1, weights=weights[0]),
Conv(input_size=(32,32,64), filter_size=(3,3,64,64), stride=2, pad1=1, pad2=1, weights=weights[1]),

Conv(input_size=(16,16,64), filter_size=(3,3,64,128), stride=1, pad1=1, pad2=1, weights=weights[2]),
Conv(input_size=(16,16,128), filter_size=(3,3,128,128), stride=2, pad1=1, pad2=1, weights=weights[3]),

Conv(input_size=(8,8,128), filter_size=(3,3,128,256), stride=1, pad1=1, pad2=1, weights=weights[4]),
Conv(input_size=(8,8,256), filter_size=(3,3,256,256), stride=2, pad1=1, pad2=1, weights=weights[5]),
]

model = Model(layers=layers)
pe, pe_maps = model.cut(params=params)
network = Network(ops=model.ops(), pe=pe, pe_maps=pe_maps)

####

x = init_x(8, (32, 32), 0, 127)
assert (np.min(x) >= 0 and np.max(x) <= 127)
y, cycles = network.forward(x=x)
# y = model.forward_dist(x=x)
y_ref = model.forward(x=x)
# print (np.shape(y), np.shape(y_ref))
# print (y[0][15][15], y_ref[0][15][15].flatten()[0:40])
assert (np.all(np.array(y) == np.array(y_ref)))
    
####

total_send = 0
total_rec = 0
total_array = 0

for pe in network.pe:
    total_send += pe.send_count
    total_rec += pe.rec_count
    total_array += len(pe.arrays)
    # print (len(pe.arrays))

print ('total pe:', len(network.pe))
print ('total array:', total_array)
print (total_send, total_rec, cycles)

####






















