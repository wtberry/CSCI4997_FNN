'''
HW3, implement fully connected 3 layaers neural networks

Modify the PATH var before running!!
'''
###VECTORIZE METHODS!: @vectorize(['float32(float32, float32)'], target='cuda'), where the things are return(param a, param b) and so on

## Libraries
import numpy as np
from numpy import vectorize
from scipy import special # for logistic function
import matplotlib.pyplot as plt
from loader import MNIST
from sklearn import preprocessing
# import scipy optimizer too??

##### 1. Import data #####
print('Loading datasets...')
PATH = '/home/wataru/Uni/4997/programming_hw/ZhuFnn/MNIST_data'
mndata = MNIST(PATH)
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

### Make one hot matrix for y (labels)
def one_hot(y):
    '''
    Return one hot matrix for label, given y matrix
    '''
    y_one_hot = np.zeros((y.shape[0], output_size))

    for i in range(y.size):
        y_one_hot[i, y[i]] = 1
    return y_one_hot

##### Feature Normalization #####
min_max_scalar = preprocessing.MinMaxScaler()
X = min_max_scalar.fit_transform(X)
X_test = min_max_scalar.fit_transform(X_test)

##### 3. Initialize #####
#Adding bias term to Xs
X = np.c_[np.ones((m_train, 1)), X]
X_test = np.c_[np.ones((m_test, 1)), X_test]

## One hotting the labels
y_label = one_hot(y)
y_test_Label = one_hot(y_test)
'''
For random samples from :math:`N(\mu, \sigma^2)`, use:

    ``sigma * np.random.randn(...) + mu`
'''
# Here, we'll initialize weights using normal distribution
sigma, mu = 0.25, 0
w1 = sigma * np.random.randn(hidden_size, input_size+1) + mu
w2 = sigma * np.random.randn(output_size, hidden_size + 1) + mu

## Maybe unroll weights into one vector to feed into the feedforward
nn_params = np.concatenate((w1.reshape(w1.size, order='F'), w2.reshape(w2.size, order='F')))

##### 4. Feedforward. #####

def sigmoid(X):
    # matrix supported elementwise sigmoid function implemented by scipy.special
    return special.expit(X)

def ReLu(X):
    return X*(X>0)

def prediction(X, w1, w2):

    '''
    feedforward and return output/prediction, given X, y, and 2 weights
    # unroll the weights back to two matrix
    '''
    # getting hidden layer nodes values
    m = X.shape[0]
    z1 = X.dot(w1.T) # shape (60000, 50)
    a1 = sigmoid(z1)
    
    # adding hidden layer bias, and getting output
    a1 = np.c_[np.ones((m, 1)), a1] # Shape of (60000, 51)
    z2 = a1.dot(w2.T) # shape of (60000, 10)
    pred = sigmoid(z2)
    
    ## converting one hot back to vector
    pred = pred.argmax(axis=1).reshape(m, 1)
    return pred

def cost(nn_params, input_size, hidden_size, output_size, X, y_one_hot, lam):

    w1 = np.reshape(nn_params[:hidden_size * (input_size + 1)], \
            (hidden_size, input_size + 1), order='F')
    w2 = np.reshape(nn_params[hidden_size * (input_size + 1):], \
             (output_size, hidden_size + 1), order='F')

    m = X.shape[0] # 60,000
    J = 0 # initializing cost

    # Initializing theta grad
    w1_grad = np.zeros((w1.shape))
    w2_grad = np.zeros((w2.shape))

    ## Feedforwarding, input to hidden
    z2 = X.dot(w1.T)
    a2 = sigmoid(z2)
    a2 = np.c_[np.ones((m, 1)), a2]

    ## Hidden to output
    z3 = a2.dot(w2.T)
    a3 = sigmoid(z3)

    J = np.sum(1/(2*m) * (a3- y_one_hot)**2)

    ## Regularization??

    ## Backprop
    ## Initializing param
    D2 = np.zeros((output_size, hidden_size+1))
    D1 = np.zeros((hidden_size, input_size+1))

    d3 = a3 - y_one_hot
    d2 = d3.dot(w2)*a2*(1-a2)

    D2 += d3.T.dot(a2)
    D1 += d2.T.dot(X)[1:, :] ## (50x785)

    w1_grad = (1/m)*D1
    w2_grad = (1/m)*D2

    grad = np.concatenate((w1_grad.reshape(w1_grad.size, order='F'), w2_grad.reshape(w2_grad.size, order='F')))

    return [J, grad]

##### Accuracy #####
def accuracy(pred, y):
    comp = pred == y
    comp = comp.astype(float)
    result = comp.mean()
    return result * 100


##### 5. Backpropagation is inside of the cost function #####

##### 6. Gradient Descent #####
sizes = {'input':input_size, 'hidden':hidden_size, 'output':output_size}

def grad_descent(X, y_label, nn_params, lr, num_iters, sizes, y):

    m = X.shape[0]
    input_size, hidden_size, output_size = sizes['input'], sizes['hidden'], sizes['output']
    j_hist = []
    lam = 0


    w1 = np.reshape(nn_params[:hidden_size * (input_size + 1)], \
            (hidden_size, input_size + 1), order='F')
    w2 = np.reshape(nn_params[hidden_size * (input_size + 1):], \
             (output_size, hidden_size + 1), order='F')

    for i in range(num_iters):

        

        J = cost(nn_params, input_size, hidden_size, output_size, X, y_label, lam)
        j_hist.append(J[0]) # storing cost function values 
        J, w_grad = J[0], J[1]
        w1_grad = np.reshape(w_grad[:hidden_size * (input_size + 1)], \
                (hidden_size, input_size + 1), order='F')
        w2_grad = np.reshape(w_grad[hidden_size * (input_size + 1):], \
                 (output_size, hidden_size + 1), order='F')

        w1 -= lr*w1_grad
        w2 -= lr*w2_grad

        if i%10 == 0:
            print('# of epoch: ', i)
            print('cost J: ', J)
        if i%100 == 0:
            pred = prediction(X, w1, w2)
            acc = accuracy(pred, y)
            print('accuracy: ', acc)
    pred = prediction(X, w1, w2)
    acc = accuracy(pred, y)
    print('Training accuracy: ', acc)
    return [w1, w2, j_hist]

##### Training #######
# printing out models params
print(hidden_size, 'hidden_nodes')
print('Learning rate: ' ,lr, 'epochs: ', epochs)

result = grad_descent(X, y_label, nn_params, lr, epochs, sizes, y)
j_hist = np.array(result[2])
w1 = result[0]
w2 = result[1]
##### 7. graphing #####
def graph_cost(j_hist):
    plt.figure()
    plt.plot(np.arange(1, j_hist.size+1), j_hist)
    plt.xlabel('# of iterations')
    plt.ylabel('cost J')
    plt.grid(True)
    plt.show()

pred_test = prediction(X_test, w1, w2)
test_acc = accuracy(pred_test, y_test)
print('Testing set accuracy: ', test_acc)
graph_cost(j_hist)
