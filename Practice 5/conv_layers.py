from builtins import range
import numpy as np


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    #https://jmyao17.github.io/Machine_Learning/Neural_Network/CNN-1/CNN_Build.html
    pad = conv_param['pad'] #variable of pad to make usage more comfortable
    stride = conv_param['stride'] #variable of stride to make usage more comfortable
    
    N, C, H, W = x.shape #assign input shapes to data points (N), channels (C), height (H)
    #and width (W)
    F, _, HH, WW = w.shape #assign filter weight shapes to filters (F), channels (_),
    # height (HH)  and width (WW)
    
    HF = 1 + (H + 2 * pad - HH) // stride #using formulas given above, result must be int
    WF = 1 + (W + 2 * pad - WW) // stride #using formulas given above, result must be int
    
    out = np.zeros((N, F, HF, WF)) #creating output data array with the shapes given above
    #filled with zeros
    
    x_pad = np.pad(x, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant') #padding the input data into new variable

    for n in range(N): #looping over data points
        for f in range(F): #looping over filters
            for hf in range(HF): #looping over heights
                for wf in range(WF): #looping over widths
                    hx = hf * stride #calculate height with stride
                    wx = wf * stride #calculate width with stride
                    out[n, f, hf, wf] = np.sum(x_pad[n, :, hx:hx+HH, wx:wx+WW]*w[f, :, :, :]) + b[f] #calculate the output by summing the padded input values that are being looped over and multiplied with the weights and finally add the bias
                    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    #https://jmyao17.github.io/Machine_Learning/Neural_Network/CNN-1/CNN_Build.html
    x, w, b, conv_param = cache #separate inputs from one variable to multiple
    
    # using some defined variables from forward pass
    
    pad = conv_param['pad'] #variable of pad to make usage more comfortable
    stride = conv_param['stride'] #variable of stride to make usage more comfortable
    
    N, C, H, W = x.shape #assign input shapes to data points (N), channels (C), height (H)
    #and width (W)
    F, _, HH, WW = w.shape #assign filter weight shapes to filters (F), channels (_),
    # height (HH)  and width (WW)
    
    HF = 1 + (H + 2 * pad - HH) // stride #using formulas given above, result must be int
    WF = 1 + (W + 2 * pad - WW) // stride #using formulas given above, result must be int 
    
    x_pad = np.pad(x, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant') #padding the input data into new variable
    
    dx_pad = np.zeros(x_pad.shape) #creating array filled with zeros to gradient x padded
    dw = np.zeros(w.shape) #creating array filled with zeros to gradient w
    db = np.zeros(b.shape) #creating array filled with zeros to gradient b
    
    for n in range(N): #looping over data points
        for f in range(F): #looping over filters
            for hf in range(HF): #looping over heights
                for wf in range(WF): #looping over widths
                    hx = hf * stride #calculate height with stride
                    wx = wf * stride #calculate width with stride                    
                    d = dout[n, f, hf, wf] #get the current value from the upstream derivatives
                    db[f] += d #update the gradient of b
                    dx_pad[n, :, hx:hx+HH, wx:wx+WW] += d * w[f] #update the gradient of x padded
                    dw[f] += d * x_pad[n, :, hx:hx+HH, wx:wx+WW] #update the gradient of the weight
    dx = dx_pad[:, :, pad:-pad, pad:-pad] #finally turn dx padded into the gradient of x needed as an ouput
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    # https://jmyao17.github.io/Machine_Learning/Neural_Network/CNN-1/CNN_Build.html
    
    N, C, H, W = x.shape #assign input shapes to data points (N), channels (C), height (H)
    #and width (W)
    stride = pool_param['stride'] #variable of stride to make usage more comfortable
    PH = pool_param['pool_height'] #variable of pooling height to make usage more comfortable
    PW = pool_param['pool_width']  #variable of pooling width to make usage more comfortable

    HF = (H - PH) // stride + 1 # calculate height output dimensions 
    WF = (W - PW) // stride + 1 # calculate width output dimensions
    
    out = np.zeros((N, C, HF, WF)) #creating output data array with the shapes given above
    #filled with zeros
    
    for n in range(N): #looping over data points
        for c in range(C): #looping over channels
            for hf in range(HF): #looping over heights
                for wf in range(WF): #looping over widths
                    hx = hf * stride #calculate height with stride
                    wx = wf * stride #calculate width with stride 
                    out[n, c, hf, wf] = np.max(x[n, c, hx:hx+PH, wx:wx+PW]) #find the max from the inputs with current indexes from loops                           
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
     # https://jmyao17.github.io/Machine_Learning/Neural_Network/CNN-1/CNN_Build.html
        
    x, pool_param = cache #separate inputs from one variable to multiple
    
    N, C, HF, WF = dout.shape #assign input shapes to data points (N), channels (C), height (H) and width (W)
    H = x.shape[2] #variable of height to make usage more comfortable
    W = x.shape[3] #variable of width to make usage more comfortable
    stride = pool_param['stride'] #variable of stride to make usage more comfortable
    PH = pool_param['pool_height'] #variable of pooling height to make usage more comfortable
    PW = pool_param['pool_width']  #variable of pooling width to make usage more comfortable
    
    # https://jmyao17.github.io/Machine_Learning/Neural_Network/CNN-1/CNN_Build.html
    
    dx = np.zeros(x.shape) #creating array filled with zeros to gradient x
    
    for n in range(N): #looping over data points
        x_prev = x[n] #use n-th sample from the inputs dataset
        for c in range(C): #looping over channels
            for hf in range(HF): #looping over heights
                for wf in range(WF): #looping over widths
                    hx = hf * stride #calculate height with stride
                    wx = wf * stride#calculate width with stride
                    x_prev_slice = x_prev[c, hx:hx+PH, wx:wx+PH] #using the current height and width corners and channel to get a slice from the current sample of inputs data
                    mask = (x_prev_slice == np.max(x_prev_slice)) #masking the slice to get the data where the max of matrix is
                    dx[n, c, hx:hx+PH, wx:wx+PH] += np.multiply(mask, dout[n, c, hf, wf]) #multiplying the mask and current data from upstream derivatives and adding it to the gradient of x variable
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

