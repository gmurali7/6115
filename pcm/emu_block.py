
import numpy as np
from conv_utils import conv_output_length

np.set_printoptions(threshold=np.inf)

###########################################

OPCODE_CONV = 1
OPCODE_DOT = 2

###########################################

def conv(x, f, b, q, stride, pad1, pad2):
    Hi, Wi, Ci = np.shape(x)
    Fh, Fw, _, Co = np.shape(f)
    Ho = conv_output_length(Hi, Fh, 'same', stride)
    Wo = conv_output_length(Hi, Fw, 'same', stride)

    x = np.pad(array=x, pad_width=[[pad1,pad2], [pad1,pad2], [0,0]], mode='constant')

    p = np.zeros(shape=(Ho, Wo, Fh * Fw * Ci))
    y = np.zeros(shape=(Ho, Wo, Co))
    
    f_matrix = np.reshape(f, (Fh * Fw * Ci, Co))

    for h in range(Ho):        
        for w in range(Wo):
            patch = np.reshape(x[h*stride:(h*stride+Fh), w*stride:(w*stride+Fw), :], -1)
            assert(np.prod(np.shape(patch)) == np.shape(f_matrix)[0])
            p[h, w, :] = patch
            y[h, w, :] = conv_kernel(patch, f_matrix, b, q)

    return y, p

def conv_kernel(patch, f, b, q):
    y = patch @ f
    assert(np.all(np.absolute(y) < 2 ** 15))
    y = y + b
    y = y * (y > 0)
    y = y.astype(int)
    # y = np.bitwise_and(y, 15)
    # quant cannot be 1
    y = y // q 
    y = np.clip(y, 0, 127)
    return y

def dot(x, w, b, q):
    y = x @ w
    assert(np.all(np.absolute(y) < 2 ** 15))
    y = y + b
    # y = y * (y > 0)
    y = y.astype(int)
    # y = np.bitwise_and(y, 15)
    # quant cannot be 1
    y = y // q
    y = np.clip(y, -128, 127)
    return y, x

###########################################

def emu(path):
    params = np.load('%s/emu.npy' % (path), allow_pickle=True).item()

    ######################

    patch = [[None for col in range(params['num_layer'])] for row in range(params['num_example'])] 
    yout  = [[None for col in range(params['num_layer'])] for row in range(params['num_example'])] 

    for ii in range(params['num_example']):
        for jj in range(params['num_layer']):
            
            param = params[jj]
        
            #######
            
            xin = params['x'][ii] if (jj == 0) else yout[ii][jj-1]
            # xin = params['x'][ii] if (jj == 0) else np.bitwise_and(yout[ii][jj-1].astype(int), 15)
            xin = np.reshape(xin, param['x'])
            
            #######

            if param['op'] == OPCODE_CONV: 
                y, p = conv(xin, param['weights'], param['bias'], param['quant'], param['dims']['stride'], param['dims']['pad1'], param['dims']['pad2'])

                shape_y = np.shape(y)
                yout[ii][jj] = np.reshape(y, (shape_y[0]*shape_y[1], shape_y[2]))
                
                shape_p = np.shape(p)
                patch[ii][jj] = np.reshape(p, (shape_p[0]*shape_p[1], shape_p[2]))
                
            else:
                yout[ii][jj], patch[ii][jj] = dot(xin, param['weights'], param['bias'], param['quant'])

    ######################

    for ii in range(params['num_example']):
        for jj in range(params['num_layer']):
            np.savetxt("%s/emu_patch%d_%d.csv" % (path, ii+1, jj+1), patch[ii][jj], fmt='%d', delimiter=" ")
            np.savetxt("%s/emu_yout%d_%d.csv" % (path, ii+1, jj+1), yout[ii][jj], fmt='%d', delimiter=" ")

    ######################




















