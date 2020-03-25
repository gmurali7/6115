
import math
import numpy as np
np.set_printoptions(threshold=1000)

from dot import *

##############################################

class Network:
    def __init__(self, cores, num_layers):
        self.cores = cores
        self.num_cores = len(cores)
        self.num_layers = num_layers

    def forward(self, x):
        num_examples, _, _, _ = np.shape(x)
        
        y = [None] * num_examples
        for example in range(num_examples):
            y[example] = x[example]
            for layer in range(self.num_layers):
                h, w, c = np.shape(y[example])
                if c <= 3:
                    y[example] = self.cores[0].forward(layer=layer, x=y[example])
                else:
                    assert ((c % self.num_cores) == 0)
                    cut_c = c // self.num_cores

                    for core in range(self.num_cores):
                        # we need to move quantization outside of layer.
                        c1 = core * cut_c
                        c2 = (core + 1) * cut_c
                        self.cores[core].forward(layer=layer, x=y[example][:, :, c1:c2])

                    y[example] = self.reduce()

    def reduce(self):
        reduce_steps = np.log2(self.num_cores)
        assert ((reduce_steps % 1) <= 0)
        reduce_steps = int(reduce_steps)
        
        for step in range(1, reduce_steps + 1):
            group = 2 ** step
            for core in range(0, self.num_cores, group):
                accum_core = core
                reduce_core = core + group // 2
                self.cores[accum_core].accum(self.cores[reduce_core].reduce())
                
        return self.cores[0].reduce()
        
##############################################

class Core:
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

class Model:
    def __init__(self, layers):
        self.layers = layers

    def cut(self, num_cores):
        num_layers = len(self.layers)
        layers = [[None for layer in range(num_layers)] for core in range(num_cores)]

        for layer in range(num_layers):
            cuts = self.layers[layer].cut(num_cores=num_cores)
            for core in range(num_cores):
                layers[core][layer] = cuts[core]

        cores = [None] * num_cores
        for core in range(num_cores):
            cores[core] = Core(layers=layers[core])

        return Network(cores=cores, num_layers=num_layers)

##############################################

class Layer:
    def __init__(self):
        assert(False)
        
    def forward(self, x):   
        assert(False)

    def rpr(self):
        assert(False)

    def cut(self, num_cores):
        assert (False)
        
##############################################

class Conv(Layer):
    def __init__(self, input_size, filter_size, stride, pad1, pad2, weights=None):

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
                
        if (self.fh == 1): 
            assert((self.s==1) and (self.p1==0) and (self.p2==0))

        maxval = 2 ** (8 - 1)
        minval = -1 * maxval
        if weights == None:
            values = np.array(range(minval + 1, maxval))
            self.w = np.random.choice(a=values, size=self.filter_size, replace=True).astype(int)
            self.b = np.zeros(shape=self.fn).astype(int)
            self.q = 200
        else:
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

    def forward(self, x):
        y = conv(x=x, f=self.w, b=self.b, q=self.q, stride=self.s, pad1=self.p1, pad2=self.p2)
        return y

    def cut(self, num_cores):
        cores = [None] * num_cores
        for core in range(num_cores):
            if self.c <= 3:
                cores[0] = self
            else:
                assert ((self.c % num_cores) == 0)
                cut_c = self.c // num_cores
                cores[core] = Conv(input_size=(self.h, self.w, cut_c), filter_size=(self.fh, self.fw, cut_c, self.fn), stride=self.s, pad1=self.p1, pad2=self.p2)

        return cores

##############################################

class Dense(Layer):
    def __init__(self, size, weights=None):
        self.size = size
        self.isize, self.osize = self.size

        maxval = 2 ** (8 - 1)
        minval = -1 * maxval
        if weights == None:
            values = np.array(range(minval + 1, maxval))
            self.w = np.random.choice(a=values, size=self.size, replace=True).astype(int)
            self.b = np.zeros(shape=self.osize).astype(int) 
            self.q = 200
        else:
            self.w, self.b, self.q = weights
            # check shape
            assert(np.shape(self.w) == self.size)
            assert(np.shape(self.b) == (self.osize,))
            assert(np.shape(self.q) == ())
            # cast as int
            self.w = self.w.astype(int)
            self.b = self.b.astype(int)
            self.q = int(self.q)
            # q must be larger than 0
            assert(self.q > 0)

    def forward(self, x):
        x = np.reshape(x, self.isize)
        y = dot(x=x, f=self.w, b=self.b, q=self.q)
        return y

    def cut(self, num_cores):
        assert (False)


#########################
        
        
        
        
        
        
        
        
        
