
import numpy as np
from conv_utils import conv_output_length

##################################################

def conv_ref(x, f, b, q, stride, pad1, pad2):
    Hi, Wi, Ci = np.shape(x)
    Fh, Fw, _, Co = np.shape(f)
    Ho = conv_output_length(Hi, Fh, 'same', stride)
    Wo = conv_output_length(Hi, Fw, 'same', stride)

    x = np.pad(array=x, pad_width=[[pad1,pad2], [pad1,pad2], [0,0]], mode='constant')
    y = np.zeros(shape=(Ho, Wo, Co))
    f_matrix = np.reshape(f, (Fh * Fw * Ci, Co))

    for h in range(Ho):        
        for w in range(Wo):
            patch = np.reshape(x[h*stride:(h*stride+Fh), w*stride:(w*stride+Fw), :], -1)
            assert(np.prod(np.shape(patch)) == np.shape(f_matrix)[0])
            y[h, w, :] = dot_ref(patch, f_matrix, b, q)

    return y

def dot_ref(x, w, b, q):
    y = x @ w
    assert(np.all(np.absolute(y) < 2 ** 23))
    y = y + b
    y = y * (y > 0)
    y = y.astype(int)
    y = y // q 
    y = np.clip(y, 0, 127)
    return y
    
##################################################
    
def conv(x, f, b, q, stride, pad1, pad2, params):
    Hi, Wi, Ci = np.shape(x)
    Fh, Fw, _, Co, bpw = np.shape(f)
    assert (bpw == params['bpw'])
    Ho = conv_output_length(Hi, Fh, 'same', stride)
    Wo = conv_output_length(Hi, Fw, 'same', stride)

    x = np.pad(array=x, pad_width=[[pad1,pad2], [pad1,pad2], [0,0]], mode='constant')
    f_matrix = np.reshape(f, (Fh * Fw * Ci, Co, params['bpw']))
    y = np.zeros(shape=(Ho, Wo, Co))
    psum = 0

    for h in range(Ho):        
        for w in range(Wo):
            print ("(%d, %d)" % (h, w))
            patch = np.reshape(x[h*stride:(h*stride+Fh), w*stride:(w*stride+Fw), :], -1)
            y[h, w, :], p = dot(patch, f_matrix, b, q, params)
            psum += p
            # print (y[h, w, :])
            
    return y, psum

def dot(x, w, b, q, params):
    y, psum = pim_dot(x, w, params)
    assert(np.all(np.absolute(y) < 2 ** 23))
    y = y + b
    y = y * (y > 0)
    y = y.astype(int)
    y = y // q 
    y = np.clip(y, 0, 127)
    return y, psum
            
##################################################

def pim_dot(x, w, params):
    nrow, ncol, nbit = np.shape(w)
    y = np.zeros(shape=ncol)
    psum = 0
    
    for b in range(params['bpa']):
        xb = np.bitwise_and(np.right_shift(x.astype(int), b), 1)
        # print ("%d | %d/%d" % (b, np.sum(xb), np.shape(xb)[0]))

        for r1 in range(0, len(xb), params['wl']):
            r2 = min(r1 + params['wl'], len(xb))
            xbr = xb[r1:r2]
            wr = w[r1:r2]
        
            assert (params['wpb'] == (params['bl'] // params['bpw']))
            for c1 in range(0, ncol, params['wpb']):
                c2 = min(c1 + params['wpb'], ncol)
                wrc = wr[:, c1:c2]
        
                pim, p = pim_kernel(xbr, wrc, b, params)
                # assert (np.all(pim == (xbr @ wr)))
                y[c1:c2] += np.left_shift(pim.astype(int), b)
                psum += p
            
    return y, psum

##################################################

def pim_kernel(x, w, b, params):
    ishape, oshape, bpw = np.shape(w)
    assert(bpw == params['bpw'])
    w_matrix = np.reshape(w, (ishape, oshape * bpw))

    shift = 2 ** np.array(range(params['bpw']))

    wl_ptr = 0
    wl = np.zeros(len(x))
    wl_sum = np.zeros(len(x))
    wl_stride = np.zeros(len(x))
    
    y = 0
    psum = 0
    flag = False
    while wl_ptr < len(x):
        # advice = be careful about: (< vs <=), (> vs >=)
        wl[0] = x[0] & (wl_ptr <= 0)
        wl_sum[0] = x[0] & (wl_ptr <= 0)
        wl_stride[0] = (wl_sum[0] <= params['adc'])
        
        for ii in range(1, len(x)):
            if params['skip']:
                row = params['adc'] if flag else params['rpr'][b]
                wl[ii]        = (x[ii] & (wl_ptr <= ii)) & (wl_sum[ii - 1] < row)
                wl_sum[ii]    = (x[ii] & (wl_ptr <= ii)) + wl_sum[ii - 1]
                wl_stride[ii] = (wl_sum[ii] <= row) + wl_stride[ii - 1]
            else:
                # assert (params['rpr'][b] == params['adc'])
                wl[ii]        = (x[ii] & (wl_ptr <= ii)) & (ii < (wl_ptr + params['adc']))
                wl_stride[ii] = wl_ptr + params['adc']

        pdot = wl @ w_matrix
        assert (np.all(pdot >= 0))
        var = np.random.normal(loc=0., scale=params['sigma'] * np.sqrt(pdot), size=np.shape(pdot))
        var = var.astype(int)
        pdot = pdot + var
        pdot = np.clip(pdot, 0, params['adc'])
        psum += 1

        flag = params['stall'] and (not flag) and (params['rpr'][b] > params['adc']) and (np.any(pdot == params['adc']))
        if not flag:
            wl_ptr = wl_stride[-1]
            x_offset = np.sum(wl).astype(int) * params['offset']
            pdot_sum = pdot.reshape(oshape, params['bpw']) @ shift
            y += pdot_sum - x_offset
        
    return y, psum

##################################################



