from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    num_dimensions = X.shape[1]
    
    for i in range(num_train):
        
        # We first calculate the scores
        score = np.dot(X[i], W)
        
        # Once we have the score function, we scale it down by a factor C, which is the maximum score between classes for each image
        score_scaled = score - np.max(score)
        
        for j in range(num_classes):
            
            if j == y[i]:
                true_class = j
                        
        # Now we can exponentiate each score, and normalize with respect to all other scores for that image
        score_exp = np.exp(score_scaled)
        score_norm = score_exp / np.sum(score_exp)
        
        # For the loss, we are interested in extracting the score of the true class for each image i, and taking -log() of that score
        loss += -1*np.log(score_norm[true_class])
        
        ##### GRADIENT #####
        
        # Gradient with respect to *scores* is the probabilities vector, minus one for the true class 
        grad = score_norm
        grad[true_class] -= 1
        grad = grad.reshape((1, num_classes))
        
        # Gradient with respect to *weights* is backpropagated through the X matrix     
        dW += np.dot(X[i].T.reshape((num_dimensions, 1)), grad)
        
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    dW /= num_train
    dW += 2* reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]
    num_classes = W.shape[1]
    num_dimensions = X.shape[1]
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
       
    ###### LOSS ########    
        
    # We first calculate the scores
    score = np.dot(X, W)

    # Once we have the score function, we scale it down by a factor C, which is the maximum score between classes for each image
    score_scaled = score - np.max(score, axis=1).reshape((num_train, 1))

    # Now we can exponentiate each score, and normalize with respect to all other scores for that image
    score_exp = np.exp(score_scaled)
    score_norm = score_exp / np.sum(score_exp, axis=1).reshape((num_train, 1))

    # For the loss, we are interested in extracting the score of the true class for each image i, and taking -log() of that score
    nolog_loss = score_norm[range(num_train), y]
    loss_mat = -1*np.log(nolog_loss)
    
    # The loss is the average - the sum of losses across all images divided by the number of images
    loss = np.sum(loss_mat)
    loss /= num_train

    # Adding regularization
    loss += reg * np.sum(W * W)
    
    ##### GRADIENT #####

    # Gradient with respect to *scores* is the probabilities vector, minus one for the true class 
    grad = score_norm
    grad[range(num_train), y] -= 1

    # Gradient with respect to *weights* is backpropagated through the X matrix     
    dW += np.dot(X.T, grad)
        
    # Taking the average gradient
    dW /= num_train
    
    # Adding regularization
    dW += 2* reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
