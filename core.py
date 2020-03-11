
import math
import numpy as np
np.set_printoptions(threshold=1000)

from layers import *

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
                for core in range(self.num_cores):
                    y[example] += self.cores[core].forward(layer=layer, x=y[example])

        return y
        
##############################################

class Core:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, layer, x):
        return self.layers[layer].forward(x)
        
##############################################
  
        
        
        
        
        
        
        
        
