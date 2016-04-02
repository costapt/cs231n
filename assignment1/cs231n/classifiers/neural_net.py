import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

class SoftmaxGate(object):

  def forward(self, W, b, X, y):
    # Initialize the loss to zero.
    loss = 0.0

    N = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]

    # f -> (N,C)
    f = X.dot(W) + b

    # Subtract each row of fs by the maximum element for numerical stability
    f -= np.max(f, axis=1).reshape(N,1)
    # e^(X*W - k)
    f = np.exp(f)

    # Get the sum of each row
    s = np.sum(f, axis=1).reshape(N,1)
    # Normalize each row to have values that sum to 1
    prob = f / s

    loss += np.sum(-np.log(prob[np.arange(N), y].clip(min=0.00000001)))
    loss /= N

    # Compute local gradient of W
    self.dW = X.T.dot(prob)
    k = csr_matrix((np.ones(N), (np.arange(N), y)), shape=(N,C)).toarray()
    self.dW -= X.T.dot(k)
    self.dW /= N

    # Compute local gradient of b
    self.dB = np.sum(prob, axis=0)
    self.dB -= np.bincount(y)
    self.dB /= N

    # Compute local gradient of x
    self.dX = f.dot(W.T) / s
    for i, yi in enumerate(y):
        self.dX[i, :] -= W[:, yi]
    self.dX /= N

    return loss

  def backward(self):
    return self.dW, self.dX, self.dB

class ReLUGate(object):

    def forward(self, z):
        self.relu = np.maximum(0,z)
        return self.relu

    def backward(self, topgrad):
        return (self.relu > 0) * topgrad


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    h1 = X.dot(W1) + b1
    relu = ReLUGate()
    z1 = relu.forward(h1)

    # If the targets are not given then jump out, we're done
    if y is None:
      return z1.dot(W2) + b2

    softmax = SoftmaxGate()
    loss = softmax.forward(W2, b2, z1, y)
    loss += 0.5 * reg * np.sum(W2 * W2)
    loss += 0.5 * reg * np.sum(W1 * W1)

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    dW2, dZ1, dB2 = softmax.backward()
    dW2 += reg * W2
    grads['W2'] = dW2

    grads['b2'] = dB2

    dZ1 = relu.backward(dZ1)
    dW1 = X.T.dot(dZ1)
    dW1 += reg * W1
    grads['W1'] = dW1

    dB1 = np.sum(dZ1, axis=0)
    grads['b1'] = dB1
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      idx = np.random.choice(num_train, batch_size)
      X_batch = X[idx]
      y_batch = y[idx]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      dW1 = grads['W1']
      dW2 = grads['W2']
      dB1 = grads['b1']
      dB2 = grads['b2']

      self.params['W1'] -= learning_rate * dW1
      self.params['W2'] -= learning_rate * dW2
      self.params['b1'] -= learning_rate * dB1
      self.params['b2'] -= learning_rate * dB2
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    relu = ReLUGate()
    h1 = X.dot(self.params['W1']) + self.params['b1']
    z1 = relu.forward(h1)
    pred = z1.dot(self.params['W2']) + self.params['b2']
    y_pred = np.argmax(pred, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred
