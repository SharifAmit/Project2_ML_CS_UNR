import numpy as np
from perceptron import perceptron_train,perceptron_test
X1 = np.array([[0,1],[1,0],[5,4],[1,1],[3,3],[2,4],[1,6]])
Y1 = np.array([[1],[1],[0],[1],[0],[0],[0]])
W = perceptron_train(X1,Y1)
accuracy = perceptron_test(X1, Y1, W[0], W[1])
print (accuracy)

X2 = np.array([[-2,1],[1,1],[1.5,-0.5],[-2,-1],[-1,-1.5],[2,-2]])
Y2 = np.array([[1],[1],[1],[-1],[-1],[-1]])
W = perceptron_train(X2,Y2)
accuracy = perceptron_test(X2, Y2, W[0], W[1])
print (accuracy)
