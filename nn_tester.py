from nearest_neighbor import choose_K
import numpy as np

X_train = np.array([[1,5],\
              [2,6],\
              [2,7],\
              [3,7],\
              [3,8],\
              [4,8],\
              [5,1],\
              [5,9],\
              [6,2],\
              [7,2],\
              [7,3],\
              [8,3],\
              [8,4],\
              [9,5]])
Y_train = np.array([[-1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[1]])

X_test = np.array([[1,1],\
              [2,1],\
              [0,10],\
              [10,10],\
              [5,5],\
              [3,10],\
              [9,4],\
              [6,2],\
              [2,2],\
              [8,7]])
Y_test = np.array([[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1]])

best_acc, best_k = choose_K(X_train,Y_train,X_test,Y_test)
print('Best Accuracy: ' + str(best_acc) + '%')
print('Best_K: ' + str(best_k))