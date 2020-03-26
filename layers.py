
import math
import numpy as np
np.set_printoptions(threshold=1000)

from dot import *
from pim import *

##############################################

class Network:
    def __init__(self, tiles, num_layers):
        self.tiles = tiles
        self.num_tiles = len(tiles)
        self.num_layers = num_layers

    def forward(self, x):
        num_examples, _, _, _ = np.shape(x)
        
        y = [None] * num_examples
        for example in range(num_examples):
            y[example] = x[example]
            for layer in range(self.num_layers):

                h, w, c = np.shape(y[example])
                
                ################################
                
                if c <= 3:
                    y[example] = self.tiles[0].forward(layer=layer, x=y[example])
                else:
                    assert ((c % self.num_tiles) == 0)
                    cut_c = c // self.num_tiles

                    for tile in range(self.num_tiles):
                        # we need to move quantization outside of layer.
                        c1 = tile * cut_c
                        c2 = (tile + 1) * cut_c
                        self.tiles[tile].forward(layer=layer, x=y[example][:, :, c1:c2])

                    y[example] = self.reduce()
                
                ################################

                y[example] += self.tiles[0].layers[layer].b
                y[example] *= (y[example] > 0)
                y[example] = y[example] // self.tiles[0].layers[layer].q 
                y[example] = np.clip(y[example], 0, 127)
                y[example] = y[example].astype(int)

        return y
                

    def reduce(self):
        reduce_steps = np.log2(self.num_tiles)
        assert ((reduce_steps % 1) <= 0)
        reduce_steps = int(reduce_steps)
        
        for step in range(1, reduce_steps + 1):
            group = 2 ** step
            for tile in range(0, self.num_tiles, group):
                accum_tile = tile
                reduce_tile = tile + group // 2
                self.tiles[accum_tile].accum(self.tiles[reduce_tile].reduce())
                
        return self.tiles[0].reduce()
        
##############################################

class Tile:
    def __init__(self, layers):
        self.layers = layers
        self.rec_count = 0
        self.send_count = 0
        self.y = 0

    def forward(self, layer, x):
        self.rec_count += np.prod(np.shape(x))
        self.y = self.layers[layer].forward(x)
        return self.y
                
    def reduce(self):
        self.send_count += np.prod(np.shape(self.y))
        return self.y
        
    def accum(self, x):
        self.rec_count += np.prod(np.shape(x))
        self.y += x
        return self.y

##############################################

class PE:
    def __init__(self, weights):
        self.layers = layers
        self.rec_count = 0
        self.send_count = 0
        self.y = 0

    def forward(self, layer, x):
        self.rec_count += np.prod(np.shape(x))
        self.y = self.layers[layer].forward(x)
        return self.y
                
    def reduce(self):
        self.send_count += np.prod(np.shape(self.y))
        return self.y
        
    def accum(self, x):
        self.rec_count += np.prod(np.shape(x))
        self.y += x
        return self.y

##############################################

class Model:
    def __init__(self, layers):
        self.layers = layers
        
    def forward(self, x):
        num_examples, _, _, _ = np.shape(x)
        num_layers = len(self.layers)
        
        y = [None] * num_examples
        for example in range(num_examples):
            y[example] = x[example]
            for layer in range(num_layers):
                y[example] = self.layers[layer].forward_ref(x=y[example])

        return y

    def cut(self, num_tiles):
        num_layers = len(self.layers)
        layers = [[None for layer in range(num_layers)] for tile in range(num_tiles)]

        for layer in range(num_layers):
            cuts = self.layers[layer].cut()
            
        assert (False)

##############################################

class Layer:
    def __init__(self):
        assert(False)
        
    def forward(self, x):   
        assert(False)
        
    def forward_ref(self, x):   
        assert(False)

    def rpr(self):
        assert(False)

    def cut(self, num_tiles):
        assert (False)
        
##############################################

class Conv(Layer):
    def __init__(self, input_size, filter_size, stride, pad1, pad2, weights, params):

        self.input_size = input_size
        self.h, self.w, self.c = self.input_size
                
        self.filter_size = filter_size
        self.fh, self.fw, self.fc, self.fn = self.filter_size
        
        assert(self.c == self.fc)
        assert(self.fh == self.fw)

        self.s = stride
        self.p1 = pad1
        self.p2 = pad2
        
        self.y_h = (self.h - self.fh + self.s + self.p1 + self.p2) / self.s
        self.y_w = (self.w - self.fw + self.s + self.p1 + self.p2) / self.s
                
        self.params = params

        maxval = 2 ** (8 - 1)
        minval = -1 * maxval
        if weights is not None:
            self.w, self.b, self.q = weights
            assert (np.all(self.w >= minval))
            assert (np.all(self.w <= maxval))
            # check shape
            assert(np.shape(self.w) == self.filter_size)
            assert(np.shape(self.b) == (self.fn,))
            assert(np.shape(self.q) == ())
            # cast as int
            self.w = self.w.astype(int)
            self.b = self.b.astype(int)
            self.q = int(self.q)
            # q must be larger than 0
            assert(self.q > 0)
        else:
            values = np.array(range(minval + 1, maxval))
            self.w = np.random.choice(a=values, size=self.filter_size, replace=True).astype(int)
            self.b = np.zeros(shape=self.fn).astype(int)
            self.q = 200
        
        ##############################

    def forward(self, x):
        y = pim_conv(x=x, f=self.w, b=self.b, q=self.q, stride=self.s, pad1=self.p1, pad2=self.p2, params=self.params)
        return y
        
    def forward_ref(self, x):
        y = conv(x=x, f=self.w, b=self.b, q=self.q, stride=self.s, pad1=self.p1, pad2=self.p2)
        return y

    def cut(self):
        
        # nrow, nwl, wl, xb = np.shape(x)
        # nwl, wl, nbl, bl = np.shape(w) 
        # nrow, ncol = y_shape

        ########################

        arrays = []
        
        ########################
        
        w_offset = self.w + self.params['offset']
        w_matrix = np.reshape(w_offset, (self.fh * self.fw * self.fc, self.fn))
        wb = []
        for bit in range(self.params['bpw']):
            wb.append(np.bitwise_and(np.right_shift(w_matrix, bit), 1))
        wb = np.stack(wb, axis=-1)
        
        ########################
        
        nrow, ncol, nbit = np.shape(wb)
        if (nrow % self.params['rpa']):
            zeros = np.zeros(shape=(self.params['rpa'] - (nrow % self.params['rpa']), ncol, nbit))
            wb = np.concatenate((wb, zeros), axis=0)

        nrow, ncol, nbit = np.shape(wb)
        wb = np.reshape(wb, (-1, self.params['rpa'], ncol, nbit))

        nwl, wl, ncol, nbit = np.shape(wb)
        wb = np.transpose(wb, (0, 1, 3, 2))
        wb = np.reshape(wb, (nwl, self.params['rpa'], nbit * ncol))
        
        nwl, wl, ncol = np.shape(wb)
        if (ncol % self.params['bl']):
            zeros = np.zeros(shape=(nwl, self.params['rpa'], self.params['bl'] - (ncol % self.params['bl'])))
            wb = np.concatenate((wb, zeros), axis=2)

        wb = np.reshape(wb, (nwl, self.params['rpa'], -1, self.params['bl']))

        print (np.shape(wb))

        return arrays

##############################################
        
        
        
        
        
        
        
        
