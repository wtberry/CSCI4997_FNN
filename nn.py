'''
HW3, implement fully connected 3 layaers neural networks
'''
## Libraries
import numpy as np
import pandas as pd
from scipy import special # for logistic function
import matplotlib.pyplot as plt
from mnist import MNIST
# import scipy optimizer too??

##### 1. Import data #####
print('Loading datasets...')
mndata = MNIST('/home/wataru/Uni/4997/programming_hw/FNN/MNIST_data')
X, y = mndata.load_training()
X_test, y_test = mndata.load_testing()

X, y = np.array(X), np.array(y).reshape(-1, 1) # X(60,0000 x 784) y(60,0000x1)
X_test, y_test = np.array(X_test), np.array(y_test).reshape(-1, 1)

##### 2. Set up parameters #####
m_train = X.shape[0]
m_test= X_test.shape[0]
input_size = X.shape[1] # number of features on the input + 1 (bias
hidden_size = 50
output_size = np.unique(y).shape[0] # extract unique elements and count them as numbers of output labels
lr = 3e-2 # learning rate
epochs = 5000 # num of epoch


##### 3. Initialize #####
#Adding bias term to Xs
X = np.c_[np.ones((m_train, 1)), X]
X_test = np.c_[np.ones((m_test, 1)), X_test]
'''
For random samples from :math:`N(\mu, \sigma^2)`, use:

    ``sigma * np.random.randn(...) + mu`
'''
# Here, we'll initialize weights using normal distribution
sigma, mu = 0.25, 0
w1 = sigma * np.random.randn(input_size + 1, hidden_size) + mu
w2 = sigma * np.random.randn(hidden_size + 1, output_size) + mu
## Maybe unroll weights into one vector to feed into the feedforward

##### 4. Feedforward. #####

def sigmoid(X):
    # matrix supported elementwise sigmoid function implemented by scipy.special
    return special.expit(X)

def feedforward(X, w1, w2):

    '''
    feedforward and return output/prediction, given X, y, and 2 weights
    # unroll the weights back to two matrix
    '''
    # getting hidden layer nodes values
    z1 = X.dot(w1) # shape (60000, 50)
    a1 = sigmoid(z1)
    
    # adding hidden layer bias, and getting output
    a1 = np.c_[np.ones((m_train, 1)), a1] # Shape of (60000, 51)
    z2 = a1.dot(w2) # shape of (60000, 10)
    pred = sigmoid(z2)
    return pred

def cost(pred, y):
    J = 1/(2*m) * (pred - y)**2

##### 5. Backpropagation #####

##### . #####
##### . #####
