"""
Module for handling machine learning model
4 method are provided:
    1. Matrix Factorization: The main algorithm of this project
    2. Clustering: Can be used as a recommendation system (just for reference)
    3. Classification: Compare with Matrix Factorization
    4. Gradient descent: ?? -> 혹 사용 하지 않게 된다면 지워 주세요. 감사 합니다.
"""

import numpy as np


class ML:
    def __init__(self, data_handler):
        """
        :param: data: data frame
        :param: n_users: number of users
        :param: n_items: number of items
        """
        self.data_handler = data_handler

    def matrix_factorization(self, K=20, steps=10, alpha=0.001, beta=0.02):
        """
        Matrix Factorization using Gradient Descent
        :param: K: number of latent features
        :param: steps: number of steps to perform the gradient descent
        :param: alpha: learning rate
        :param: beta: regularization parameter
        :return: the final matrices P and Q
        """
        data, n_users, n_items = self.data_handler.load_data()

        # Initialize the user and item latent feature matrice
        P = np.random.normal(scale=1. / K, size=(n_users, K))
        Q = np.random.normal(scale=1. / K, size=(n_items, K))

        # Initialize the biases
        b_u = np.zeros(n_users)
        b_i = np.zeros(n_items)
        b = np.mean(data['rating'])

        # Create a list of training samples
        rows, cols = data.shape
        samples = [
            (i, j, data.iloc[i, j])
            for i in range(rows)
            for j in range(cols)
        ]

        # Perform stochastic gradient descent for number of steps
        training_process = []
        for step in range(steps):
            np.random.shuffle(samples)
            for i, j, r in samples:
                # Computer prediction and error
                prediction = b + b_u[i] + b_i[j] + P[i, :].dot(Q[j, :].T)
                e = (r - prediction)

                # Update biases
                b_u[i] += alpha * (e - beta * b_u[i])
                b_i[j] += alpha * (e - beta * b_i[j])

                # Update user and item latent feature matrices
                P[i, :] += alpha * (e * Q[j, :] - beta * P[i, :])
                Q[j, :] += alpha * (e * P[i, :] - beta * Q[j, :])

            # Compute total mean squared error
            mse = 0