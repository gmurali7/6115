
import numpy as np
from conv_utils import conv_output_length

##################################################
    
def pim_conv(x, f, b, q, stride, pad1, pad2, params):
    Hi, Wi, Ci = np.shape(x)
    Fh, Fw, _, Co = np.shape(f)
    Ho = conv_output_length(Hi, Fh, 'same', stride)
    Wo = conv_output_length(Hi, Fw, 'same', stride)

    x = np.pad(array=x, pad_width=[[pad1,pad2], [pad1,pad2], [0,0]], mode='constant')
    f_matrix = np.reshape(f, (Fh * Fw * Ci, Co)) + params['offset']
    y = np.zeros(shape=(Ho, Wo, Co))

    for h in range(Ho):        
        for w in range(Wo):
            patch = np.reshape(x[h*stride:(h*stride+Fh), w*stride:(w*stride+Fw), :], -1)
            y[h, w, :] = pim_dot(patch, f_matrix, b, q, params)
            
    return y
    
##################################################

def pim_dot(x, w, b, q, params):
    y = pim_dot_kernel(x, w, params)
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

def pim_dot_kernel(x, w, params):
    y = 0
    for x_bit in range(params['bpa']):
        xb = np.bitwise_and(np.right_shift(x.astype(int), x_bit), 1)
        for w_bit in range(params['bpw']):
            wb = np.bitwise_and(np.right_shift(w.astype(int), w_bit), 1)
            dot = xb @ wb
            y = y + np.left_shift(dot.astype(int), x_bit + w_bit)
            
        y -= (np.sum(xb) << (x_bit + 7))
            
    assert(np.all(y < 2 ** 23))
    return y

###########################################





