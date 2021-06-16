from builtins import range
from builtins import object
import numpy as np

from layers import *
from layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        # See also: http://cs231n.github.io/neural-networks-2/#init                #
        ############################################################################
        
        #https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
        self.params['W1'] = np.random.normal(scale = weight_scale, size = (input_dim, hidden_dim)) #using np random normal distibution (Gaussian), where scale is the given weight scale and size matches the input dimensions and hidden layer dimensions
        self.params['b1'] = np.zeros(hidden_dim) #assining bias zeros based on hidden layer dimensions
        self.params['W2'] = np.random.normal(scale = weight_scale, size = (hidden_dim, num_classes)) #using np random normal distibution (Gaussian), where scale is the given weight scale and size matches the hidden layer dimensions and output (num of classes) dimensions
        self.params['b2'] = np.zeros(num_classes) #assining bias zeros based on output (num of classes) dimensions
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        
        W1, b1 = self.params['W1'], self.params['b1'] #assign weights and biases for the first layer
        W2, b2 = self.params['W2'], self.params['b2'] #assign weights and biases for the second layer
        
        out_rf, cache_rf = affine_relu_forward(X, W1, b1) #using X, W1 and b1 to perform transform followed by a ReLU                         
        out_af, cache_af = affine_forward(out_rf, W2, b2) #using the previous results and W2 and b2 to perform forward transform
        
        scores = out_af                                     
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        loss, dx = softmax_loss(scores, y) #calculate softmax loss with given function
        
        loss += 0.5 * self.reg * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2)) #calculation of L2 regularization
                                             
        dx_ab, grads['W2'], grads['b2'] = affine_backward(dx, cache_af) #performing backward pass on 2nd layer, finding the gradients of dx2, dw2 and db2
        grads['W2'] += self.reg*self.params['W2'] #applyi ngregularization to dw2
        
        dx_arb, grads['W1'], grads['b1'] = affine_relu_backward(dx_ab, cache_rf) #performing the backward pass for the 1st layer, finding the gradients of dx1, dw1 and db1       
        grads['W1'] += self.reg*self.params['W1'] #applying regularization to dw1                                        
                                
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        
        dims_1 = input_dim #the dimensions of the first layer
        
        for layer in range(1, self.num_layers + 1): #looping thorugh network layer by layer
            if layer < self.num_layers: #layer num is not the last layer
                dims_2 = hidden_dims[layer - 1] #getting the dimensions of the next hidden layer
                self.params['W' + str(layer)] = np.random.normal(scale = weight_scale, size = (dims_1, dims_2)) #using np random normal distibution (Gaussian), where scale is the given weight scale and size matches the dimensions of the first and the second layer in the loop
                self.params['b' + str(layer)] = np.zeros(dims_2) #assining bias zeros based on the layer dimensions
                
            if layer == self.num_layers: # if layer is the last layer
                self.params['W' + str(layer)] = np.random.normal(scale = weight_scale, size = (dims_2, num_classes)) #using np random normal distibution (Gaussian), where scale is the given weight scale and size matches the dimensions of the last hidden layer and the output (num of classes) layer dimensions
                self.params['b' + str(layer)] = np.zeros(num_classes) #assining bias zeros based on the output (num of classes) layer
            
            dims_1 = dims_2 #change the second layer into the first layer to perform loop on the next layer
                           
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        
        scores = X # assign input values to scores
        affine_cache = {} #empty dictionary to save affine forward pass results
        relu_cache = {} #empty dictionary to save relu forward pass results
        dropout_cache = {} #empty dictionary to save dropout forward pass results
        
        for layer in range(1, self.num_layers): #looping through the layers
            if layer < self.num_layers: #if not the last layer
                scores, cache = affine_forward(scores, self.params['W' + str(layer)], self.params['b' + str(layer)]) #using scores (X), current weights and biases to perform affine forward pass transform          
                affine_cache[layer] = cache #save cache into the affine dictionary
                
                out, cache = relu_forward(scores) #using the previous score results perform transform followed by a ReLU  
                relu_cache[layer] = cache #save cache into relu dictionary
                
                scores = out #assign the latest layer output to scores
                
            if self.use_dropout == True: #check if dropout is used
                scores, cache = dropout_forward(scores, self.dropout_param) #perform dropout forward pass using the scores and dropout parameter
                dropout_cache[layer] = cache #save cache into dropout dictionary
        
        scores, last_cache = affine_forward(scores, self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)]) #using scores the final forward scores, weights and biases to perform affine forward pass transform  on the last layer 
        
        if self.use_dropout == True:#check if dropout is used
            scores, cache = dropout_forward(scores, self.dropout_param)  #perform dropout forward pass using the final scores and dropout parameter on the last layer
            dropout_cache[layer] = cache #save cache into dropout dictionary

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        loss, dx = softmax_loss(scores, y) #find loss and dx using the given softmax loss function
        
        for layer in range(1, self.num_layers + 1): #looping through the layers
            loss +=  0.5 * self.reg * np.sum(self.params['W' + str(layer)]**2) #adding L2 regularization to the current layer
        
        dx, dw, db = affine_backward(dx, last_cache) #perfom affine backward pass using the dx and the last cache data to get gradients dx, dw and db for the last layer
        grads['W' + str(self.num_layers)] = dw + self.reg * self.params['W' + str(self.num_layers)] #add regularization to the last weight gradients
        grads['b' + str(self.num_layers)] = db #update the last bias gradients
        

        for layer in range(self.num_layers - 1, 0, -1): #looping through the layers backwards
        ##could not properly implement
            ##if self.use_dropout == True:
            ##    dx = dropout_backward(dx, dropout_cache[layer])
                
            dx = relu_backward(dx, relu_cache[layer]) #update dx using the relu backward pass with current dx and data from the relu cache dictonary saved during the forward pass
            dx, dw, db = affine_backward(dx, affine_cache[layer]) #perform affine backward pass using the updated dx and data from the affine cache dictionary
            grads['W' + str(layer)] = dw + self.reg * self.params['W' + str(layer)] #updating the current layer weight gradients and adding regularization
            grads['b' + str(layer)] = db #updating the current layer biases
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
