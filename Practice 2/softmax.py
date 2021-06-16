import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax (cross-entropy) loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    You might or might not want to transform it into one-hot form (not obligatory)
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  # In this naive implementation we have a for loop over the N samples
  for i, x in enumerate(X):
        
    #############################################################################
    # TODO: Compute the cross-entropy loss using explicit loops and store the   #
    # sum of losses in "loss".                                                  #
    # If you are not careful in implementing softmax, it is easy to run into    #
    # numeric instability, because exp(a) is huge if a is large.                #
    #############################################################################
    # TODO: should use explicit loops here
    z = x.dot(W) #calculate z based on the formula z = xW
    p = np.exp(z) / np.sum(np.exp(z)) #calculating the normalized probabilities
    pc = p[y[i]] #probability of the true/correct label
    loss += - np.log(pc) #finding, adding the cross-entropy loss for true label
    
    #############################################################################
    # TODO: Compute the gradient using explicit loops and store the sum over    #
    # samples in dW.                                                            #
    #############################################################################
    
    # source https://cs231n.github.io/neural-networks-case-study/
    for j in range(W.shape[1]): # looping over 10 classes
        if j == y[i]: # checking whether the class is same or not
            dW[:, j] += (np.exp(z[y[i]]) / np.sum(np.exp(z)) - 1) * x
            # in case of the correct class, substract -1 for negative influence on
            # the loss
        else:
            dW[:, j] += np.exp(z[j]) / np.sum(np.exp(z)) * x
            # in other cases, perform basic calculus based on the gradient formula
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
  # now we turn the sum into an average by dividing with N
  loss /= X.shape[0]
  dW /= X.shape[0]

  # Add regularization to the loss and gradients.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax (cross-entropy) loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the cross-entropy loss and its gradient using no loops.     #
  # Store the loss in loss and the gradient in dW.                            #
  # Make sure you take the average.                                           #
  # If you are not careful with softmax, you migh run into numeric instability#
  #############################################################################
  
  z = X.dot(W) #calculate z based on the formula z = xW
  
  p = np.exp(z) / np.sum(np.exp(z), axis = 1, keepdims = True) #calculate proba-
  # bilities normalized, denominator is summed by axis 1 and keeping the dimensions
  
  loss = np.average(-np.log(p[range(X.shape[0]), y]))  #finding the probability
  # from p based on y value, finding logs and calculating their average for final loss
  
  # source https://cs231n.github.io/neural-networks-case-study/
  dscores = p #determining new variable for gradient on the scores 
  dscores[range(X.shape[0]), y] -= 1 #subtract -1 from the probs to have negative
  # influence on the loss
  dscores /= X.shape[0]
  dW = np.dot(X.T, dscores) #finding product of X and the gradient based on the formula
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  # Add regularization to the loss and gradients.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W

  return loss, dW

