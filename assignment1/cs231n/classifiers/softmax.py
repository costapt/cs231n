import numpy as np
from random import shuffle

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

  N = X.shape[0]
  C = W.shape[1]

  for xi, yi in zip(X,y):
      f = xi.dot(W)
      f -= np.max(f) # for numerical stability.
      f = np.exp(f)

      s = np.sum(f)
      prob = f[yi] / s
      loss += -np.log(prob)

      dW[:, yi] += xi * prob - xi
      for j in range(C):
          if j == yi:
              continue

          dW[:, j] += xi * f[j] / s

  loss /= N
  loss += 0.5 * reg * np.sum(W * W)

  dW /= N
  dW += reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]

  # f -> (N,C)
  f = X.dot(W)

  # Subtract each row of fs by the maximum element for numerical stability
  f -= np.max(f, axis=1).reshape(N,1)
  # e^(X*W - k)
  f = np.exp(f)

  # Get the sum of each row
  s = np.sum(f, axis=1).reshape(N,1)
  # Normalize each row to have values that sum to 1
  prob = f / s

  loss += np.sum(-np.log(prob[np.arange(N), y].clip(min=0.00000001)))

  dW = X.T.dot(prob)
  # How do I vectorize this? dW[:,y] -= X.T does not work
  for xi, yi in zip(X,y):
      dW[:, yi] -= xi

  loss /= N
  aux = 0.5 * reg * W
  loss += np.sum(aux * aux)

  dW /= N
  dW += reg * W
  return loss, dW
