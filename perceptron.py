"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        #self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.
        Initialize self.w as a matrix with random numbers in the range [0, 1)

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        N, n_dim = X_train.shape[0], X_train.shape[1] 
        self.w = np.random.rand(self.n_class, n_dim) # (C, D)
        losses = []
        
        for i in range(self.epochs):
            
            p = np.random.permutation(N)
            X_temp, y_temp = X_train[p], y_train[p]
            
            #iterating through the dataset
            for n in range(N):
                xi = X_temp[n] #1,D
                yi = y_temp[n] #1,1
                
                #score
                score = np.dot(xi, self.w.T) #(1, D) * (D, C) -> (1, C)
                
                #pred label based on current weight
                pred_label = np.argmax(score)
                
                #update Wc
                to_update_c = score > score[yi]
                
                if yi != pred_label:
                    self.w[yi] += self.lr * xi
                    self.w[to_update_c] -= self.lr * xi
            
            self.lr = self.lr * 0.95

        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        test_score = np.dot(X_test, self.w.T) #(N, C)
        return np.argmax(test_score, axis=1)
