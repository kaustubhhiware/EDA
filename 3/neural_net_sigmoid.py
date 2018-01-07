import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import recall_score, classification_report, confusion_matrix
from time import sleep
import pickle

TRAIN = int(0)
TEST = int(1)
EXIT = int(2)

weights = {}
bias = {}

numY = 0
numFeatures = 0
alpha = 0.001  # rate for change
##
# Gradient descent using backpropogation
# Members :
# 14CS10003 - Akhil Jain
# 14CS10028 - Krishang Garodia
# 14CS10050 - Vaibhav Agarwal
# 14CS10054 - Yara Manisha
# 14CS30011 - Kaustubh Hiware
##
valid_input = {0, 1, 2}  # internal variable, for input


def get_input():
    """
        get correct input from user in while loop
    """
    while(1):
        mode = raw_input('+--- Press 0 to Train, 1 to test, 2 to exit : ')
        if int(mode) not in valid_input:
            print '+--- Incorrect input'
            continue
        else:
            return int(mode)


def getdata(mode):
    """
        read data from file and return
        numY: Number of Y labels
        numFeatures: Number of features in X
    """

    if mode == TRAIN:
        filename = 'data-train.txt'
    else:
        filename = 'data-test.txt'

    filer = open(filename, 'r')
    text = filer.read()
    data = text.split('\n')

    X = []
    Y = []
    for each in data:
        if each == '':
            continue
        l = [float(i) for i in each.split(' ')]
        X += [l[1:-1]]
        Y += [int(l[-1])]

    filer.close()
    global numY, numFeatures
    numY = len(set(Y))  # number of distinct labels in Y
    numFeatures = len(X[0])

    return X, Y


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(x):
    """
        return differential for sigmoid
    """
    ##
    # D  (   e^x   )  =   (   e^x   )   -  (   e^x   )^2
    # -   ---------        ---------        ---------
    # Dx (1 + e^x)        (1 + e^x)        (1 + e^x)^2
    ##
    return x * (1 - x)


def train(X, Y, nodes, layers, iters):
    """
        How to train your dragon
        layers = Number of layers expect only input layer
    """
    np.random.seed(0)  # like SRAND(TIME(NULL))
    # set random normalised weights (w) and bias (b)
    w = []  # use a separate var to keep changing var
    b = []
    #
    # p_in => Input layer to first hidden layer
    # p_hid => Within hidden layers
    # p_out => Hidden layer to output layer
    #
    # make sure -1 < w < 1
    w_in = 2 * np.random.randn(numFeatures, nodes) - 1
    w_hid = 2 * np.random.randn(nodes, nodes) - 1
    w_out = 2 * np.random.randn(nodes, numY) - 1
    # all b's set to 0
    b_in = np.zeros((1, nodes))
    b_hid = np.zeros((1, nodes))
    b_out = np.zeros((1, numY))

    w.append(w_in)
    w.append((layers-1) * w_hid)
    w.append(w_out)  # append for unequal size
    b.append(b_in)
    b.append((layers-1) * b_hid)
    b.append(b_out)

    print '+-------+-----------+'
    print '| Iters |    Loss   |'
    print '+-------+-----------+'

    for i in range(iters):
        # https://iamtrask.github.io/2015/07/27/python-network-part2/

        # front propogation in play
        layer = [None] * (layers+1)
        layer[0] = X

        for j in range(layers):
            # k = layer[j].dot(w[j])
            # k += b[j]
            # layer[j+1] = sigmoid(k)
            layer[j+1] = sigmoid(layer[j].dot(w[j]) + b[j])

        # start back propogation
        error = [None] * (layers+1)
        delta = [None] * (layers+1)

        error[layers] = layer[layers] - Y.reshape((len(Y), 1))
        delta[layers] = error[layers] * dsigmoid(layer[layers])
        # only first output layer defined non-recursively
        for k in reversed(range(layers)):
            # use differential update rule to get updated value
            error[k] = delta[k+1].dot(np.transpose(w[k]))
            delta[k] = error[k] * dsigmoid(layer[k])

            # apply gradient descent now
            w[k] = w[k] - alpha * (np.transpose(layer[k]).dot(delta[k+1]))
            # b[k] = b[k] - alpha * (delta[k])

        # print loss occasionally
        if i % 1000 == 0:
            loss = float(np.mean(np.abs(error[layers])))
            x = ''
            if loss < 10:
                x = ' '

            # print('Loss after iteration %i: %f' % (i, loss))
            print '|', "%05d" % i, '|', x, "{:.5f}".format(loss), '|'

    print '+-------+-----------+'
    return w, b


def predict(X):
    """
        predict function
    """
    layers = len(weights) - 1
    layer = [None] * (layers+1)
    layer[0] = X
    # print weights

    print 'Layers:', layers
    for i in range(layers):
        layer[i+1] = sigmoid(layer[i].dot(weights[i]) + bias[i])

    # Return class with max probability
    return np.argmax(layer[layers], axis=1)


def main():
    """
        Create a neural net to train and test GD-BP
    """
    while(1):
        mode = get_input()

        if mode == TRAIN:
            print 'Train'
            X, Y = getdata(mode)
            print 'Output labels:', numY
            print 'number of features:', numFeatures

            # split data, needed for cross validation
            # size of S1 and S2 kept same, 2*l/3
            X_s1 = np.array(X[: 2*len(X)/3])
            X_s2 = np.array(X[len(X)/3:])
            Y_s1 = np.array(Y[: 2*len(Y)/3])
            Y_s2 = np.array(Y[len(Y)/3:])

            print 'Enter number of nodes in hidden layer,  hidden layers,  iterations/passes: '
            nodes = raw_input()
            nodes, layers, iters = [int(i) for i in nodes.split(" ")]
            global weights, bias
            weights, bias = train(X_s1, Y_s1, nodes, layers, iters)

            Y_out = predict(X_s2)
            print 'Accuracy:', accuracy_score(Y_s2, Y_out)
            print classification_report(Y_s2, Y_out)

            Y_out = predict(X_s1)
            print 'Accuracy:', accuracy_score(Y_s1, Y_out)
            print classification_report(Y_s1, Y_out)

            print 'Saving lists to file'
            with open('weights', 'wb') as filename:
                pickle.dump(weights, filename)
            with open('bias', 'wb') as filename:
                pickle.dump(bias, filename)

        elif mode == TEST:
            print 'Test'
            X, Y = getdata(mode)
            X = np.array(X)
            Y = np.array(Y)

            print 'Reading from files'
            with open('weights', 'rb') as filename:
                global weights
                weights = pickle.load(filename)
            with open('bias', 'rb') as filename:
                global bias
                bias = pickle.load(filename)

            Y_out = predict(X)
            print 'Accuracy:', accuracy_score(Y, Y_out)
            print classification_report(Y, Y_out)

        else:
            exit(0)


if __name__ == '__main__':
    main()
