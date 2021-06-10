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
    
    cantEjemplos = X.shape[0]
    cantClases = W.shape[1]
    L = np.zeros(cantEjemplos)
    
    for i in range(cantEjemplos):
        # valor de temperatura para experimentar para la pregunta 2
        T = 1
        #calculo las probabilidades de cada clase para el ejemplo i
        #obtengo vector de probabilidades de tamaño c
        score = np.dot(X[i], W)
        estabilizador = - np.max(score)
        probabilidades = np.exp((score + estabilizador)/T) / np.sum(np.exp((score + estabilizador)/T), axis=0)

        #calculo L[i] como -log de la probabilidad de la clase del ejemplo
        #Agrego en la matriz de gradiente en cada columna (clase) X[i]*prob[c] si c no es la clase objetivo
        #sino  X[i]*prob[c] -1 si c es la clase objetivo
        for c in range(cantClases):
            if c == y[i]:
                L[i] = -np.log(probabilidades[c])
                dW[:,c] = dW[:,c] + X[i]*(probabilidades[c] -1)
            else:
                dW[:,c] = dW[:,c] + X[i]*(probabilidades[c])
    
    #la pérdida será el promedio de L, más la regularización
    loss = np.sum(L)/cantEjemplos + 0.5 * reg * np.sum(W*W)
    #la matriz de gradiente debe ser dividida por la cantidad de ejemplos y se le agrega la regularización
    dW = dW/cantEjemplos + reg*W
    

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    cantEjemplos = X.shape[0]
    
    # valor de temperatura para experimentar para la pregunta 2
    T = 1
        
    #calculo las probabilidades de cada clase para todos los ejemplos
    #obtengo vector una matriz de probabilidades de tamaño n x c
    score = np.dot(X, W)
    estabilizador = - np.max(score)
    probabilidades = np.exp((score + estabilizador)/T)
    vector_normalizador = np.sum(probabilidades, axis=1)
    probabilidades = np.divide(probabilidades, vector_normalizador.reshape(cantEjemplos, 1))
    
    # Para cada ejemplo i se obtiene probabilidad[i,y[i]] y se le calcula -log. 
    # Luego la pérdida será el promedio de dichos valores, más la regularización
    loss = np.sum(-np.log(probabilidades[np.arange(cantEjemplos), y]))/cantEjemplos + 0.5 * reg * np.sum(W*W)
    
    #inicializo la matriz de gradiente como la matriz de las probabilidades
    dW = probabilidades
    # para cada i se obtiene dW[i,y[i]]. Para esos casos el valor en dW será la probabilidad[i,y[i]] -1. 
    # como ya se había inciialiado la matriz con el valor de las probabilidades, basta con restarle uno a dichas celdas
    dW[np.arange(cantEjemplos), y] -= 1 
    #la matriz de gradiente debe ser dividida por la cantidad de ejemplos y se le agrega la regularización
    dW = np.dot(X.T, dW)/cantEjemplos + reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
