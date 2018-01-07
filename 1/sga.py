# SGA LR model binary classification
##
## Kaustubh Hiware
## @kaustubhhiware
##
from math import exp, log # e^

def log_10(x):
    log_10 = log(x) / log(10)
    return log_10


def main():

    with open("data.txt") as f:
        lines = f.read().splitlines()

    data = list()
    for each in lines:
        data.append([float(i) for i in each.split("\t")])

    theta = [0.0, 0.04, -0.02, 0.034]
    r = 0.5
    num_iterations = 2000 # maximum upto 2000 iters
    delta = 0.002 # set a value for convergence,
    # when log(L(0)) changes by less than this, we have achieved convergence
    log = 0
    print "The final values can be found at the end"
    print "i\t theta0\t\t theta1\t\t theta2\t\t theta3\t\t Error"
    for i in xrange(0,num_iterations):

        for each in data:# for each sample

            SoF = theta[0] + theta[1]*each[1] + theta[2]*each[2] + theta[3]*each[3]
            LR_SoF = exp(SoF) / (1 + exp(SoF))
            y = each[4]

            theta[0] = theta[0] + r * (y - LR_SoF)
            for j in xrange(1,4):
                theta[j] = theta[j] + r * (y - LR_SoF) * each[j]

        # find error in function
        log_l = 0
        for each in data:
            y = each[4]
            SoF = theta[0] + theta[1]*each[1] + theta[2]*each[2] + theta[3]*each[3]
            LR = exp(SoF) / (1 + exp(SoF))
            log_l += y * log_10(LR) + (1-y) * log_10(1-LR)

        print i,
        for j in xrange(0,4):
            print "\t", theta[j],
        print "\t", log_l

        if(abs(log - log_l) < delta):
            break
        log = log_l

    print "\nA convergence of ", delta, "is achieved",
    print "after", i, "iterations.Final values : "
    for each in theta:
        print each, "\t",
    print
    print "at a likelihood of", log


if __name__ == '__main__':
    main()
