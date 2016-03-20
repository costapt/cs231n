import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]
        dW[:,y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_train
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_train = X.shape[0]
  num_classes = W.shape[1]

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # N x C -> score of each image into each class.
  sj = X.dot(W)

  # N x 1 -> score of the correct class
  si = sj[np.arange(num_train),y].reshape((num_train, 1))

  # max(0, sj - si + 1) score of each class in relation to the correct one
  # inside a margin. The distance between the score of the incorrect class and
  # the correct one must be greater than a margin (in this case margin=1 but it
  # could be any number greater than 0). When sj - si + 1 is less than 0, this
  # means that the model is already correctly separating the two classes
  margin = np.maximum(sj - si + 1, 0)

  # Sum the errors of each image
  # subtract one for the comparison with itself. We did max(0, si - si + 1)
  # which equals 1 for every image.
  summed_margin = np.sum(margin, axis=1) - 1

  loss += np.sum(summed_margin) / num_classes
  # Add the regularization term
  loss += 0.5 * reg * np.sum(W * W)

  margin[margin > 0] = 1
  # For each training example, count how many classes got a better
  # classification (or did not get a proper margin) than the true class.
  # We subtract one for the comparison with itself (same as comment above).
  num_errors = np.sum(margin, axis=1) - 1

  # Num_errors is the number of times we need to subtract Xi for the gradient.
  margin[np.arange(num_train),y] = -num_errors
  dW = X.T.dot(margin)

  dW /= num_train
  dW += reg * W

  return loss, dW
