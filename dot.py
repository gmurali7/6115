
import numpy as np
from conv_utils import conv_output_length

##################################################

def conv(x, f, b, q, stride, pad1, pad2):
    Hi, Wi, Ci = np.shape(x)
    Fh, Fw, Fc, Co = np.shape(f)
    assert (Ci == Fc)
    Ho = conv_output_length(Hi, Fh, 'same', stride)
    Wo = conv_output_length(Hi, Fw, 'same', stride)

    x = np.pad(array=x, pad_width=[[pad1,pad2], [pad1,pad2], [0,0]], mode='constant')
    y = np.zeros(shape=(Ho, Wo, Co))
    f_matrix = np.reshape(f, (Fh * Fw * Ci, Co))

    for h in range(Ho):        
        for w in range(Wo):
            patch = np.reshape(x[h*stride:(h*stride+Fh), w*stride:(w*stride+Fw), :], -1)
            assert(np.prod(np.shape(patch)) == np.shape(f_matrix)[0])
            y[h, w, :] = dot(patch, f_matrix, b, q)

    return y

def dot(x, w, b, q):
    y = x @ w
    assert(np.all(np.absolute(y) < 2 ** 23))
    '''
    y = y + b
    y = y * (y > 0)
    y = y.astype(int)
    y = y // q 
    y = np.clip(y, 0, 127)
    '''
    return y
    
##################################################



