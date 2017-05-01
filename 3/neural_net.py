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
rate = 0.001  # rate for change
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
    return (np.exp(x) / (1.0 + np.exp(x)))


def train(X, Y, nodes, layers, iters):
    """
        How to train your dragon
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
    w_in = np.random.randn(numFeatures, nodes) / np.sqrt(numFeatures)
    w_hid = np.random.randn(nodes, nodes) / np.sqrt(nodes)
    w_out = np.random.randn(nodes, numY) / np.sqrt(nodes)
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
        # page 14/29
        # http://www.cedar.buffalo.edu/~srihari/CSE574/Chap5/Chap5.3-BackProp.pdf
        # https://stats.stackexchange.com/questions/65977/the-tanh-activation-function-in-backpropagation
        a = {}
        z = {}
        d = {}   # store diff
        dw = {}
        db = {}
        z[0] = X.dot(w[0]) + b[0]  # Sum(WiXi) + B0, dot is dot product
        a[0] = np.tanh(z[0])

        for j in range(layers - 1):
            z[j+1] = a[j].dot(w[j+1]) + b[j+1]
            a[j+1] = np.tanh(z[j+1])

        z[layers] = a[layers-1].dot(w[layers]) + b[layers]

        e_z = np.exp(z[layers])  # e^z
        # sum over each vector
        sigma = np.sum(e_z, axis=1, keepdims=True)
        p = e_z / sigma  # Probability

        d[layers] = p
        d[layers][range(len(X)), Y] -= 1

        # from reference, still a bit unclear ^^ srihari's
        # PART BELOW HERE NEEDS TO BE READ FROM THE REFERENCE TO UNDERSTAND
        dw[layers] = (np.transpose(a[layers-1])).dot(d[layers])
        db[layers] = np.sum(d[layers], axis=0, keepdims=True)

        # back propogation: Start from second last layer
        # and go all the way back to input layer
        for j in range(layers-2, -1, -1):
            d[j+1] = d[j+2].dot(np.transpose(w[j+2])) * (1 - np.power(a[j+1], 2))  # Ignore PEP8Bear
            dw[j+1] = np.dot(np.transpose(a[j]), d[j+1])
            db[j+1] = np.sum(d[j+1], axis=0)

        d[0] = d[1].dot(np.transpose(w[1])) * (1 - np.power(a[0], 2))
        dw[0] = np.dot(np.transpose(X), d[0])
        db[0] = np.sum(d[0], axis=0)

        for j in range(layers+1):
            dw[j] += rate * w[j]
            w[j] += -rate * dw[j]
            b[j] += -rate * db[j]

        if i % 1000 == 0:
            # calculate loss

            z = {}
            a = {}
            z[0] = X.dot(w[0]) + b[0]
            a[0] = np.tanh(z[0])

            for i in range(layers-1):
                z[i+1] = a[i].dot(w[i+1]) + b[i+1]
                a[i+1] = np.tanh(z[i+1])

            z[layers] = a[layers-1].dot(w[layers]) + b[layers]

            e_z = np.exp(z[layers])  # e^z
            # sum over each vector
            sigma = np.sum(e_z, axis=1, keepdims=True)
            p = e_z / sigma  # Probability

            logP = -np.log(p[range(len(X)), Y])
            diff = np.sum(logP)

            squares_sigma = 0.00
            for j in range(layers+1):
                squares_sigma += np.sum(np.square(w[j]))

            diff += rate / 2 * (squares_sigma)
            loss = 1.0 / len(X) * diff
            x = ''
            if loss < 10:
                x = ' '
            # print('Loss after iteration %i: %f' % (i, loss))
            print '|', "%05d" % i, '|', x, "{:.5f}".format(loss), '|'

    print '+-------+-----------+'
    return w, b


def test(X):
    """
        predict function
    """
    z = {}
    a = {}
    z[0] = X.dot(weights[0]) + bias[0]
    a[0] = np.tanh(z[0])

    for i in range(len(weights) - 2):
        z[i+1] = a[i].dot(weights[i+1]) + bias[i+1]
        a[i+1] = np.tanh(z[i+1])

    z[len(weights) - 1] = a[len(weights) - 2].dot(weights[len(weights)-1]) + bias[len(weights) - 1]  # Ignore PEP8Bear

    e_z = np.exp(z[len(weights) - 1])
    p = e_z / np.sum(e_z, axis=1, keepdims=True)
    return np.argmax(p, axis=1)  # Return class with max probability


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

            Y_out = test(X_s2)
            print 'Accuracy:', accuracy_score(Y_s2, Y_out)
            print classification_report(Y_s2, Y_out)

            Y_out = test(X_s1)
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

            Y_out = test(X)
            print 'Accuracy:', accuracy_score(Y, Y_out)
            print classification_report(Y, Y_out)

        else:
            exit(0)


if __name__ == '__main__':
    main()
