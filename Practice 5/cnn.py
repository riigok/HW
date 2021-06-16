from builtins import object
import numpy as np

from layers import *
from fast_layers import *
from layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        #https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
        self.params['W1'] = np.random.normal(scale = weight_scale, size = (num_filters, input_dim[0], filter_size, filter_size)) #using np random normal distibution (Gaussian), where scale is the given weight scale and size matches the num of filters, num of channel, and num of filter sizes
        self.params['W2'] = np.random.normal(scale = weight_scale, size = (int(num_filters * input_dim[1]/2 * input_dim[2]/2), hidden_dim)) #using np random normal distibution (Gaussian), where scale is the given weight scale and size matches the the row size (computed by multiplying num of filters, half height and width data, must be integer) and hidden layer dimensions
        self.params['W3'] = np.random.normal(scale = weight_scale, size = (hidden_dim, num_classes)) #using np random normal distibution (Gaussian), where scale is the given weight scale and size matches the hidden layer dimensions and output (num of classes) dimensions
                                             
        self.params['b1'] = np.zeros(num_filters) #assining bias zeros based on num of filters
        self.params['b2'] = np.zeros(hidden_dim) #assining bias zeros based on hidden layer dimensions
        self.params['b3'] = np.zeros(num_classes)#assining bias zeros based on output (num of classes) dimensions                                 
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        conv_relu_out, conv_relu_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param) #1st layer forward - conv relu pool 
        #forward with the arguments of input data, weight1, bias 1, convulolutional parameters and pooling parameters
        affine_relu_out, affine_relu_cache = affine_relu_forward(conv_relu_out, W2, b2)#2nd layer forward - affine relu forward
        # with the arguments of previous layer results, W2 and b2
        affine_out, affine_cache = affine_forward(affine_relu_out, W3, b3)#3rd and final layer forward - affine forward with
        #the arguments of previous layer results, W3 and b3
        scores = affine_out #assign the last results into scores variable
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dx = softmax_loss(scores, y) #calculate loss and gradient of x with softmax loss
        
        for layer in range(1, 4): #looping through the layers
            loss +=  0.5 * self.reg * np.sum(W1**2) #adding L2 regularization to the current layer
        
        affine_dx, affine_dw, affine_db = affine_backward(dx, affine_cache) #3rd layer backward - affine backward with the
        # arguments of gradient of x and the cache from the forward pass
        affine_relu_dx, affine_relu_dw, affine_relu_db = affine_relu_backward(affine_dx, affine_relu_cache) #2nd layer backward -
        # affine relu backward with the arguments from previous results and cache from the forward pass
        conv_relu_dx, conv_relu_dw, conv_relu_db = conv_relu_pool_backward(affine_relu_dx, conv_relu_cache) #1st layer backward -
        # conv relu pool backward with the arguments from previous results and chache from the forward pass
           
        
        grads['W1'] = conv_relu_dw + self.reg * W1 #updating the W1 with results from backward pass
        grads['W2'] = affine_relu_dw + self.reg * W2 #updating the W2 with results from backward pass 
        grads['W3'] = affine_dw + self.reg * W3 #updating the W3 with results from backward pass
        grads['b1'] = conv_relu_db #updating the b1 W3 with results from backward pass
        grads['b2'] = affine_relu_db #updating the b2 W3 with results from backward pass
        grads['b3'] = affine_db #updating the b3 W3 with results from backward pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
