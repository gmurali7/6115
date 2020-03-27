
import math
import numpy as np
np.set_printoptions(threshold=1000)

from dot import *
from pim import *

##############################################

class Network:
    def __init__(self, ops, arrays, array_maps):
        self.ops = ops
        self.arrays = arrays
        self.array_maps = array_maps
        self.num_layers = len(array_maps)
        self.cycles = 0

    def forward(self, x):
        num_examples, _, _, _ = np.shape(x)
        
        y = [None] * num_examples
        for example in range(num_examples):
            y[example] = x[example]
            for layer in range(min(example, self.num_layers) + 1):
                print ('%d: layer: %d example: %d' % (example, layer, example - layer))
                y[example - layer] = self.conv(layer=layer, x=y[example - layer])
                
        return y
        
    def conv(self, x, layer):
        op = self.ops[layer]
        b, q = op['b'], op['q']
        H, W, C = op['h'], op['w'], op['c']
        K, N = op['k'], op['n']
        S, P1, P2 = op['s'], op['p1'], op['p2']
        
        Ho = conv_output_length(H, K, 'same', S)
        Wo = Ho
        Co = N
                
        x = np.pad(array=x, pad_width=[[P1,P2], [P1,P2], [0,0]], mode='constant')
        y = np.zeros(shape=(Ho, Wo, Co))

        ad, ah, aw, _ = np.shape(self.array_maps[layer])
        for pix in range(Ho * Wo):
            h = pix // Wo
            w = pix % Wo
            a = pix % ad
            patch = np.reshape(x[h*S:(h*S+K), w*S:(w*S+K), :], -1)
            
            for i in range(ah):
                for j in range(aw):
                    for bit in range(8):
                        x1 = i * 128
                        x2 = min(x1 + 128, len(patch))
                        
                        y1 = j * 16
                        y2 = y1 + 16

                        xb = np.bitwise_and(np.right_shift(patch[x1:x2].astype(int), bit), 1)

                        array, partition = self.array_maps[layer][a][i][j]
                        self.arrays[array].dot(partition, xb, bit)
                        
                    y[h, w, y1:y2] += self.arrays[array].reduce()

        y = y + b
        y = y * (y > 0)
        y = y.astype(int)
        y = y // q 
        y = np.clip(y, 0, 127)
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
'''
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
'''
##############################################

'''
array level design questions:
1) A) start with N arrays B) create a kernel and then figure out duplication 
2) A) init an array with list of weights B) or program them in.

> we are doing neither right now.
'''

class Array:
    def __init__(self, weights, params):
        self.rec_count = 0
        self.send_count = 0
    
        self.weights = weights
        self.params = params
        self.shift = 2 ** np.array(range(self.params['bpw']))

        self.y = 0

    def dot(self, partition, x, x_bit):
        assert (np.all( (x == 0) + (x == 1) ))
        self.rec_count += 1 # np.prod(np.shape(x))
        pprod = x @ self.weights[0:len(x), :]
        pprod = np.reshape(pprod, (-1, self.params['bpw'])) @ self.shift
        pprod = np.left_shift(pprod.astype(int), x_bit)
        offset = 128 * np.sum(np.left_shift(x, x_bit))
        self.y += (pprod - offset)
        # self.send_count += 1 # np.prod(np.shape(y))
        # return pprod - offset

    def reduce(self):
        self.send_count += 1 # np.prod(np.shape(self.y))
        ret = self.y 
        self.y = 0
        return ret
        
    '''
    def accum(self, x):
        self.rec_count += np.prod(np.shape(x))
        self.y += x
        return self.y
    '''

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
                y[example] = self.layers[layer].forward(x=y[example])

        return y
        
    def forward_dist(self, x):
        num_examples, _, _, _ = np.shape(x)
        num_layers = len(self.layers)
        
        y = [None] * num_examples
        for example in range(num_examples):
            y[example] = x[example]
            for layer in range(num_layers):
                y[example] = self.layers[layer].forward_dist(x=y[example])

        return y

    def ops(self):
        ret = []
        for layer in self.layers:
            ret.append(layer.op())

        return ret

    def cut(self, params):
        num_layers = len(self.layers)
        
        nmac = 0
        weights = []
        for layer in range(num_layers):
            nmac += self.layers[layer].nmac
            weights.append(self.layers[layer].cut(params=params))

        arrays = []
        array_maps = []
        for layer in range(num_layers):
            p = self.layers[layer].nmac / nmac
            ndup = p * (2048 * 128 * 128) / np.prod(np.shape(self.layers[layer].cut(params=params)))
            ndup = int(ndup)
            
            nwl, _, nbl, _ = np.shape(weights[layer])
            array_map = np.zeros(shape=(ndup, nwl, nbl, 2), dtype=np.int32) 
            
            for dup in range(ndup):
                for wl in range(nwl):
                    for bl in range(nbl):
                        arrays.append(Array(weights=weights[layer][wl, :, bl, :], params=params))
                        array_map[dup][wl][bl] = np.array([len(arrays) - 1, 0])

            array_maps.append(array_map)
            
        for layer in range(num_layers):
            self.layers[layer].set_arrays(arrays)
            self.layers[layer].set_array_maps(array_maps[layer])
            
        return arrays, array_maps
                    
##############################################

class Layer:
    def __init__(self):
        assert(False)
        
    def forward(self, x):   
        assert(False)
        
    def rpr(self):
        assert(False)

    def cut(self, params):
        assert (False)
        
##############################################

class Conv(Layer):
    def __init__(self, input_size, filter_size, stride, pad1, pad2, weights):

        self.input_size = input_size
        self.xh, self.xw, self.xc = self.input_size
                
        self.filter_size = filter_size
        self.fh, self.fw, self.fc, self.fn = self.filter_size
        
        assert(self.xc == self.fc)
        assert(self.fh == self.fw)

        self.s = stride
        self.p1 = pad1
        self.p2 = pad2
        
        self.yh = (self.xh - self.fh + self.s + self.p1 + self.p2) / self.s
        self.yw = (self.xw - self.fw + self.s + self.p1 + self.p2) / self.s
        
        self.nmac = (self.fh * self.fw * self.fc * self.fn) * (self.yh * self.yw)
        # self.cells = (self.fh * self.fw * self.fc * self.fn) * 8
        # self.cells = np.prod(np.shape(self.wb)) # not the above.

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
        
    def set_arrays(self, arrays):
        self.arrays = arrays

    def set_array_maps(self, array_maps):
        self.array_maps = array_maps

    ##############################

    def forward(self, x):
        y = conv(x=x, f=self.w, stride=self.s, pad1=self.p1, pad2=self.p2)

        y = y + self.b
        y = y * (y > 0)
        y = y.astype(int)
        y = y // self.q 
        y = np.clip(y, 0, 127)

        return y

    def forward_dist(self, x):
        Ho = conv_output_length(self.xh, self.fh, 'same', self.s)
        Wo = Ho
        Co = self.fn

        x = np.pad(array=x, pad_width=[[self.p1,self.p2], [self.p1,self.p2], [0,0]], mode='constant')
        y = np.zeros(shape=(Ho, Wo, Co))

        ad, ah, aw, _ = np.shape(self.array_maps)

        for pix in range(Ho * Wo):
            h = pix // Wo
            w = pix % Wo
            a = pix % ad
            patch = np.reshape(x[h*self.s:(h*self.s+self.fh), w*self.s:(w*self.s+self.fw), :], -1)
            
            for i in range(ah):
                for j in range(aw):
                    x1 = i * 128
                    x2 = min(x1 + 128, len(patch))
                    
                    y1 = j * 16
                    y2 = y1 + 16

                    # print (x1, x2, y1, y2)

                    array, partition = self.array_maps[a][i][j]
                    y[h, w, y1:y2] += self.arrays[array].dot(partition, patch[x1:x2])


        y = y + self.b
        y = y * (y > 0)
        y = y.astype(int)
        y = y // self.q 
        y = np.clip(y, 0, 127)

        return y

    ##############################
    
    def op(self):
        ret = {'b': self.b, 'q': self.q, 'h': self.xh, 'w': self.xw, 'c': self.xc, 'k': self.fh, 'n': self.fn, 's': self.s, 'p1': self.p1, 'p2': self.p2}
        return ret
        
    ##############################

    def cut(self, params):
        
        # nrow, nwl, wl, xb = np.shape(x)
        # nwl, wl, nbl, bl = np.shape(w) 
        # nrow, ncol = y_shape

        ########################

        w_offset = self.w + params['offset']
        w_matrix = np.reshape(w_offset, (self.fh * self.fw * self.fc, self.fn))
        wb = []
        for bit in range(params['bpw']):
            wb.append(np.bitwise_and(np.right_shift(w_matrix, bit), 1))
        wb = np.stack(wb, axis=-1)
        
        ########################
        
        nrow, ncol, nbit = np.shape(wb)
        if (nrow % params['rpa']):
            zeros = np.zeros(shape=(params['rpa'] - (nrow % params['rpa']), ncol, nbit))
            wb = np.concatenate((wb, zeros), axis=0)

        nrow, ncol, nbit = np.shape(wb)
        wb = np.reshape(wb, (-1, params['rpa'], ncol, nbit))
        
        ########################

        nwl, wl, ncol, nbit = np.shape(wb)
        wb = np.reshape(wb, (nwl, params['rpa'], ncol * nbit))
        
        nwl, wl, ncol = np.shape(wb)
        if (ncol % params['bl']):
            zeros = np.zeros(shape=(nwl, params['rpa'], params['bl'] - (ncol % params['bl'])))
            wb = np.concatenate((wb, zeros), axis=2)

        wb = np.reshape(wb, (nwl, params['rpa'], -1, params['bl']))

        ########################

        return wb
        
        

##############################################
        
        
        
        
        
        
        
        
