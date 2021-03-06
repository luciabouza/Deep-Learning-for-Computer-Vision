from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    z = np.dot(prev_h, Wh) + np.dot(x, Wx) + b
    next_h = np.tanh(z)    
    cache = (x, prev_h, Wx, Wh, b, z)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, prev_h, Wx, Wh, b, z = cache
    
    # derivada de y=tan(z) respecto a z
    dyRespectZ = 1 - np.tanh(z)**2
    #regla de la cadena dL/dz = dL/dy * dy/dz. donde dL/dy=dnext_h, el input
    dLRespectZ = dnext_h * dyRespectZ
    
    #ahora aplicamos regla de la cadena para cada parametro: dL/dParametro = dL/dz * dz/dParametro
    # esa dz/dParametro es facil porque z es lineal. 
    
    dx =  np.dot(dLRespectZ, Wx.T)
    dprev_h =  np.dot(dLRespectZ, Wh.T)
    dWx =  np.dot(x.T, dLRespectZ)
    dWh =  np.dot(prev_h.T, dLRespectZ)
    db =   np.sum(dLRespectZ, axis=0)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    (N, T, D) = x.shape
    H = b.shape[0]
    prev_h = h0
    h = np.zeros((N,T,H))
    
    cache = list()     

    for t in range(T):
        next_h, cacheh = rnn_step_forward(x[:,t,:], prev_h, Wx, Wh, b) 
        h[:,t,:] = next_h
        prev_h = next_h
        cache.append(cacheh)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 
    
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # saco tama??os
    (N, T, H) = dh.shape 
    (x0, prev_h, Wx, Wh, b, z) = cache[0]
    D = x0.shape[1]
    
    # inicializo vectores a devolver
    dx = np.zeros((N,T,D))
    dWx = np.zeros((D,H))
    dWh = np.zeros((H,H))
    db = np.zeros(H)
    
    # incicializo primer vector de gradiente de h que se propaga.
    # para el ??ltimpo instante de tiempo deber?? ser cero ya que no trae info de ning??n instante posterior    
    dprev_ht = np.zeros((N,H))

    for t in range(T-1,-1,-1):
        cachenext = cache[t]
        
        # considero los gradientes de L en el instante de tiempo t (input) y tambi??n el gradeinte de h del instante posterior
        dhnext = dh[:,t,:] + dprev_ht
        dxt, dprev_ht, dWxt, dWht, dbt = rnn_step_backward(dhnext, cachenext) 
        
        # sumo todas las contribuciones al gradiente de Wx, Wh y b
        dWx += dWxt
        dWh += dWht
        db += dbt
        
        # acutalizo el gradiente en funci??n de la entrada para ese instante de tiempo
        dx[:,t,:] = dxt
        
    # el ??ltimo gradiente de h del instante anterior, como itero con desde el ultimo instante, ser?? el dh0   
    dh0 = dprev_ht


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    (N, T) = x.shape 
    (V, D) = W.shape 
    out = np.zeros((N,T,D))
    for n in range(N):
        for t in range(T):
            out[n][t] = W[x[n][t]]

    cache = x, T, V, D
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    (x, T, V, D) = cache
    
    dW = np.zeros((V,D))
    np.add.at(dW, x, dout)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = x >= 0
    neg_mask = x < 0
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    H = prev_h.shape[1]

    A = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
    i = sigmoid(A[:, 0:H])
    f = sigmoid(A[:, H:2*H])
    o = sigmoid(A[:, 2*H:3*H])
    g = np.tanh(A[:, 3*H:4*H])
    
    next_c = f * prev_c + i * g
    next_h = o * np.tanh(next_c)
  
    cache = (x, prev_h, prev_c, Wx, Wh, b, i , f, o, g, next_c, A)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    (x, prev_h, prev_c, Wx, Wh, b, i , f, o, g, next_c, A) = cache
    
    ### primer paso del grafo computacional #####
    
    dnext_h_Respect_o = dnext_h * np.tanh(next_c)
    
    dnext_h_Respect_next_c = dnext_h * (1 - np.tanh(next_c)**2) * o
    
    dnext_c_completa = dnext_c + dnext_h_Respect_next_c
    
    ### segundo paso del grafo computacional ###
        
    dnext_c_Respect_f = dnext_c_completa * prev_c
    dnext_c_Respect_i = dnext_c_completa * g
    dnext_c_Respect_g = dnext_c_completa * i
    dnext_c_Respect_prev_c = dnext_c_completa * f
    
    ### tercer paso del grafo computacional ###
    
    H = dnext_h.shape[1]
    A_1erasH = A[:, 0:H]
    A_2dasH = A[:, H:2*H]
    A_3erasH = A[:, 2*H:3*H]
    A_4tasH = A[:, 3*H:4*H]
    
    dA_1erasH_Respect_i = sigmoid(A_1erasH) * (1 - sigmoid(A_1erasH)) * dnext_c_Respect_i
    dA_2dasH_Respect_f = sigmoid(A_2dasH) * (1 - sigmoid(A_2dasH)) * dnext_c_Respect_f
    dA_3erasH_Respect_o = sigmoid(A_3erasH) * (1 - sigmoid(A_3erasH)) * dnext_h_Respect_o
    dA_4tasH_Respect_g = (1 - np.tanh(A_4tasH)**2) * dnext_c_Respect_g
    
    dA =  np.concatenate((dA_1erasH_Respect_i,dA_2dasH_Respect_f,dA_3erasH_Respect_o, dA_4tasH_Respect_g), axis=1)
    
    ### cuarto paso grafo computacional. gradientes respecto a variables ###
    
    dx = np.dot(dA, Wx.T)
    dprev_h = np.dot(dA, Wh.T)
    dprev_c = dnext_c_Respect_prev_c
    dWx = np.dot(x.T, dA)
    dWh = np.dot(prev_h.T, dA)
    db = np.sum(dA, axis=0)
    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    (N, T, D) = x.shape
    H = h0.shape[1]
    prev_h = h0
    prev_c = np.zeros(prev_h.shape)
    h = np.zeros((N,T,H))
    
    cache = list()     
    
    for t in range(T):
        next_h, next_c, cacheh = lstm_step_forward(x[:,t,:], prev_h, prev_c, Wx, Wh, b) 
        h[:,t,:] = next_h
        prev_h = next_h
        prev_c = next_c
        cache.append(cacheh)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # saco tama??os
    (N, T, H) = dh.shape 
    (x0, prev_h, prev_c, Wx, Wh, b, i , f, o, g, next_c, A) = cache[0]
    D = x0.shape[1]
    
    # inicializo vectores a devolver
    dx = np.zeros((N,T,D))
    dh0 = np.zeros((N,H))
    dWx = np.zeros((D,4*H))
    dWh = np.zeros((H,4*H))
    db = np.zeros(4*H)
    
    # incicializo primer vector de gradiente de h que se propaga.
    # para el ??ltimpo instante de tiempo deber?? ser cero ya que no trae info de ning??n instante posterior    
    dprev_ht = np.zeros((N,H))
    dprev_ct = np.zeros((N,H))

    for t in range(T-1,-1,-1):
        cachenext = cache[t]
        
        # considero los gradientes de L en el instante de tiempo t (input) y tambi??n el gradeinte de h del instante posterior
        dhnext = dh[:,t,:] + dprev_ht
        dcnext = dprev_ct
        dxt, dprev_ht, dprev_ct, dWxt, dWht, dbt = lstm_step_backward(dhnext, dcnext, cachenext)
        
        # sumo todas las contribuciones al gradiente de Wx, Wh y b
        dWx += dWxt
        dWh += dWht
        db += dbt
        
        # acutalizo el gradiente en funci??n de la entrada para ese instante de tiempo
        dx[:,t,:] = dxt
        
    # el ??ltimo gradiente de h del instante anterior, como itero con desde el ultimo instante, ser?? el dh0   
    dh0 = dprev_ht

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose:
        print("dx_flat: ", dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
