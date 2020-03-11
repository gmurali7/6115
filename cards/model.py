
import numpy as np
np.set_printoptions(threshold=1000)

class model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        num_examples, _, _, _ = np.shape(x)
        num_layers = len(self.layers)
        
        y = [None] * num_examples
        psum = np.zeros(shape=(num_examples, num_layers))
        for example in range(num_examples):
            y[example] = x[example]
            for layer in range(num_layers):
                y[example], p = self.layers[layer].forward(x=y[example])
                psum[example][layer] += p

        return y, psum

    


        
        
        
        
        
        
        
        
