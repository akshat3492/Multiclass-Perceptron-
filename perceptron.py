"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        self.w = None 
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        n_class = self.n_class
        n_iters = self.epochs
        lr = self.lr
        self.w = np.random.rand(n_class, X_train.shape[1])
        
        #vectorized form - faster
        for iter in range(n_iters):
            print("epoch ", iter)
            for i in range(len(X_train)):
                X_temp = X_train[i]#1, D
                label = y_train[i]
                predict = np.dot(w, X_temp.T)#(C,D) * (D,1) -> (C,1)
                predict_temp = np.argmax(np.dot(w, X_temp.T))
                updates_c = [predict > predict[label]]
                if label!=predict_temp:
                    self.w[label] = self.w[label] + lr*X_temp
                    self.w[tuple(updates_c)] = self.w[tuple(updates_c)] - lr*X_temp
   
        #indidual one/non vectorized - slower
        for iter in range(n_iters):
            print("epoch ", iter)
            for i in range(len(X_train)):
                X_temp = X_train[i]#1, D
                label = y_train[i]
                true_pred = np.dot(w[label], X_temp.T)
                for j in range(len(n_class)):
                    predict = np.dot(w[j], X_temp.T)#(1,D) * (D,1) -> (1,1)
                    if predict > true_pred:
                        self.w[label] = self.w[label] + lr*X_temp
                        self.w[j] = self.w[j] - lr*X_temp
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        pred = np.argmax(np.dot(w, X_test.T), axis = 1) #(C,D) * (D,N) -> (C,N)
        return
