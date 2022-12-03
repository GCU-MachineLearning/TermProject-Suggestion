"""
Module for handling machine learning model
4 method are provided:
    1. Matrix Factorization: The main algorithm of this project
    2. Clustering: Can be used as a recommendation system (just for reference)
    3. Classification: Compare with Matrix Factorization
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

    def matrix_factorization(self, K=20, steps=10001, alpha=0.001, beta=0.02, save: bool = False):
        """
        Matrix Factorization using Gradient Descent
        :param: K: number of latent features
        :param: steps: number of steps to perform the gradient descent
        :param: alpha: learning rate
        :param: beta: regularization parameter
        :return: the final matrices P and Q
        """
        data, n_users, n_items = self.data_handler.load_data()
        test_mse_lst = []
        train_mse_lst = []

        # Initialize the user and item latent feature matrices
        P = np.random.normal(scale=1. / K, size=(n_users + 1, K))  # n_users + 1 for the bias term
        Q = np.random.normal(scale=1. / K, size=(n_items + 1, K))  # n_items + 1 for the bias term

        # Initialize the biases
        b_u = np.zeros(n_users + 1)
        b_i = np.zeros(n_items + 1)
        b = np.mean(data['rating'])

        # Create a list of training samples
        # contains (user_id, item_id, rating)
        rows, cols = data.shape  # 100'000, 4
        samples = [
            (data.iloc[i, 0], data.iloc[i, 1], data.iloc[i, 2])
            for i in range(rows)
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

            for i, j, r in samples:
                prediction = b + b_u[i] + b_i[j] + P[i, :].dot(Q[j, :].T)
                mse += (r - prediction) ** 2

            mse /= len(samples)
            training_process.append((step, mse))

            if (step % 10) == 0:
                print("Iteration: %d ; error = %.4f" % (step, mse))

                mf_mse = self.test_matrix_factorisation(P, Q, b_u, b_i, b)
                test_mse_lst.append(mf_mse)
                train_mse_lst.append(mse)
                print("TEST MSE - MF MSE: {}".format(mf_mse))

        print(training_process)

        if save:
            self.save_model(P, Q, b_u, b_i, b, iteration=steps)
            self.draw_plot(test_mse_lst, train_mse_lst)
        
        return P, Q, b_u, b_i, b, training_process

    def draw_plot(self, test_mse_lst, train_mse_lst):
        from matplotlib import pyplot as plt

        plt.plot(test_mse_lst, label='test_mse', color='red')
        plt.plot(train_mse_lst, label='train_mse', color='blue')
        plt.xlabel('iteration: x10')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()

    def save_model(self, P, Q, b_u, b_i, b, iteration):
        """
        Save model
        :param P, Q, b_u, b_i, b: model parameter
        :param iteration: number of iteration
        :return: NaN
        """

        np.save(f'./pretrained_model/P_{iteration}.npy', P)
        np.save(f'./pretrained_model/Q_{iteration}.npy', Q)
        np.save(f'./pretrained_model/b_u_{iteration}.npy', b_u)
        np.save(f'./pretrained_model/b_i_{iteration}.npy', b_i)
        np.save(f'./pretrained_model/b_{iteration}.npy', b)

    def load_model(self, iteration):
        """
        Load pretrained-model parameter
        :param iteration: iteration number
        :return: trained_model parameter
        """

        # check if the model path exists
        import os
        if not os.path.exists('pretrained_model'):
            raise Exception('The model path does not exist. Please train the model first.')

        P = np.load(f'./pretrained_model/P_{iteration}.npy')
        Q = np.load(f'./pretrained_model/Q_{iteration}.npy')
        b_u = np.load(f'./pretrained_model/b_u_{iteration}.npy')
        b_i = np.load(f'./pretrained_model/b_i_{iteration}.npy')
        b = np.load(f'./pretrained_model/b_{iteration}.npy')

        return P, Q, b_u, b_i, b

    def test_matrix_factorisation(self, P, Q, b_u, b_i, b):
        """
        :param: P: user latent feature
        :param: Q: item latent feature
        :param: b_u: user bias
        :param: b_i: item bias
        :param: b: global bias
        :return: prediction
        """
        # Create a list of test samples
        # contains (user_id, item_id, rating)
        test_data = self.data_handler.load_test()

        rows, cols = test_data.shape
        samples = [
            (test_data.iloc[i, 0], test_data.iloc[i, 1], test_data.iloc[i, 2])
            for i in range(rows)
        ]

        # Compute MSE for the test samples
        mse = 0
        for i, j, r in samples:
            prediction = b + b_u[i] + b_i[j] + P[i, :].dot(Q[j, :].T)
            mse += (r - prediction) ** 2

        mse /= len(samples)

        return mse

    def movie_suggestion_mf(self, user_id, parameter_list: list):
        """
        Function that recommend movie to specific user X.
        If moive K has high expectation rating of user X -> Can be regarded that user X likes movie K.

        :param: user_id: user id
        :param: item_id: item
        :param: parameter_list: parameter list
        :return: prediction
        """
        P, Q, b_u, b_i, b = parameter_list

        # load_test
        test_data = self.data_handler.load_test()
        test_data = self.data_handler.load_test()

        rows, cols = test_data.shape
        samples = [
            (test_data.iloc[i, 0], test_data.iloc[i, 1], test_data.iloc[i, 2])
            for i in range(rows)
        ]

        movie_score = []
        item_name = self.data_handler.load_item()

        for _, j, r in samples:
            prediction_score = b + b_u[user_id] + b_i[j] + P[user_id, :].dot(Q[j, :].T)
            movie_name = item_name.iloc[j, 1]
            # if movie_name is not exist in movie_score
            if movie_name not in [x[0] for x in movie_score]:
                movie_score.append((movie_name, f'{prediction_score:.3f}'))

        movie_score.sort(key=lambda x: x[1], reverse=True)
        return movie_score

    def classification(self, c_type) :
        """
        Use various classification model
        :param: c_type: type of classification model
        :return: trained model only
        """
        # Load training data
        train_data, _, _ = self.data_handler.load_data()
        train_data = train_data.drop(columns=['timestamp'])
        train_data = train_data.values
        
        if c_type == 'knn':
            from sklearn.neighbors import KNeighborsClassifier

            # Create KNN classifier
            knn = KNeighborsClassifier(n_neighbors=5)

            # Train the classifier
            knn.fit(train_data[:, 0:2], train_data[:, 2])

            return knn  # return trained model

        elif c_type == 'decisionTree':
            from sklearn.tree import DecisionTreeClassifier

            # Create DecisionTree Classifier
            decisionT = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=0)

            # Train the classifier
            decisionT.fit(train_data[:, 0:2], train_data[:, 2])

            return decisionT  # return trained model

        elif c_type == 'svm':
            from sklearn.svm import SVR

            # Create DecisionTree Classifier
            svm = SVR(kernel='rbf', C=0.1, gamma = 0.5)

            # Train the classifier
            svm.fit(train_data[:, 0:2], train_data[:, 2])
            return svm  # return trained model
        
        elif c_type == 'randomForest':
            from sklearn.ensemble import RandomForestClassifier
            
            # Create DecisionTree Classifier
            r_forest= RandomForestClassifier(criterion='entropy', bootstrap=True, random_state=42, max_depth=5)

            # Train the classifier
            r_forest.fit(train_data[:, 0:2], train_data[:, 2])
            return r_forest  # return trained model
            
        else:
            raise ValueError('Invalid type of classification model')

    def test_classification(self, trained_model, c_type):
        """
        :param: trained_model: trained model
        :param: c_type: type of classification model
        :return: accuracy
        """
        assert trained_model is not None, 'trained model is None'
        if c_type == 'knn' or c_type =='decisionTree' or c_type == 'svm' or c_type =='randomForest' :
            # Load test data
            test_data = self.data_handler.load_test()
            test_data = test_data.drop(columns=['timestamp'])
            test_data = test_data.values

            predictions = trained_model.predict(test_data[:, 0:2])
            mse = np.sum((predictions - test_data[:, 2]) ** 2) / len(test_data)

            return mse

        else:
            raise ValueError('Invalid type of classification model')
