"""
Module for handling machine learning model
4 method are provided:
    1. Matrix Factorization: The main algorithm of this project
    2. Clustering: Can be used as a recommendation system (just for reference)
    3. Classification: Compare with Matrix Factorization
"""

"""
<< 지울 주석 >>
전체 흐름을 보기 위해서 matrix factorization 함수를 우선 작성 하였 습니다.
이 matrix factorization 함수는 user_id, item_id, rating 을 통해 학습 하고, 
user_id, item_id 에 따른 rating 을 예측 하도록 우선 구현 되어 있습 니다. 

matrix_factorization() >> 학습하고, 학습된 파라미터를 반환합니다.
test_matrix_factorization() >> MSE 를 계산합니다.
movie_suggestion_mf() >> ratings 순서대로 영화를 추천 합니다.

(N 이라는 사용자에 대해서, X 라는 영화에 대한 평점을 예측함으로써, N에게 X 라는 영화가 추천(= 높은 평점) 할지, 아닐지(= 낮은 평점)를 판단합니다.)

구현 된 함수들은 제가 틀을 짜기 위해 느낌 보려고 작성 한 것이기 때문에, 자유롭게 날리시거나, 수정해주시면 감사할 것 같습니다. 
main.py 함수에서 보셨듯, 이렇게 학습된 rating 과 다양한 classifier 를 통해 예츠된 rating 을 비교 하고 있습 니다.
때문에, 다양한 classification 방법을 추가 하여, 간단히 비교하는 것도 좋아 보입니다. 
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

    def matrix_factorization(self, K=20, steps=100, alpha=0.001, beta=0.02):
        """
        Matrix Factorization using Gradient Descent
        :param: K: number of latent features
        :param: steps: number of steps to perform the gradient descent
        :param: alpha: learning rate
        :param: beta: regularization parameter
        :return: the final matrices P and Q
        """
        data, n_users, n_items = self.data_handler.load_data()

        # Initialize the user and item latent feature matrices
        """ << 지울 주석 >>
        n_user + 1 한 이유는, user_id 가 1부터 시작 해서 해당 부분 처리 해주기 위해 진행 했습 니다. 
        """
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

            """ << 지울 주석 >>
            아래 과정은, matrix factorization 을 위한 gradient descent 를 구현 한 것 입니다.
            gradient descent 를 구현 하는 부분이 필요 한지, 혹은 sklearn 을 사용할 지 proposal 단계 에서
            명확히 확정 되지 않은 것 같아, 일단은 이렇게 구현 해 두었 습니다.
            
            추후 코드 작성 과정 에서 필요 하지 않다고 판단 되시면, 그냥 라이브러리 사용 해 주시면 될 것 같습 니다. 
            정확도 는 iteration = 200 기준(나머지 파라미터 기본 설정 유지 시), train_mse = 0.5277 입니다. 
            일반 적인 노트북 에서 약 2~3 분 정도 소요 됩니다. 
            """
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

        return P, Q, b_u, b_i, b, training_process

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
        특정 사용자 X 에 대해서 영화 추천을 해주는 함수 입니다.
        X 에 대해 K 라는 영화의 예상 rating 이 높다면 -> X 가 K 를 좋아할 것이라고 예상 할 수 있습니다.

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







    def library_matrix_factorization(self):
        """
        :param:
        :return:
        """
        # from implicit.als import AlternatingLeastSquares
        pass

    def clustering(self):
        """
        :param: 추가 하실 파라미터 에 대한 간략한 설명
        :return: return 값에 대한 간략한 설명
        """
        pass

    def classification(self, c_type: str = 'knn'):
        """
        Use various classification model
        :param: c_type: type of classification model
        :return: trained model only
        """
        if c_type == 'knn':
            from sklearn.neighbors import KNeighborsClassifier

            # Load training data
            train_data, _, _ = self.data_handler.load_data()
            train_data = train_data.drop(columns=['timestamp'])
            train_data = train_data.values

            # Create KNN classifier
            knn = KNeighborsClassifier(n_neighbors=5)

            # Train the classifier
            knn.fit(train_data[:, 0:2], train_data[:, 2])

            return knn  # return trained model

        elif c_type == 'other_model':
            pass

        else:
            raise ValueError('Invalid type of classification model')

    def test_classification(self, trained_model, c_type: str = 'knn'):
        """
        :param: trained_model: trained model
        :param: c_type: type of classification model
        :return: accuracy
        """
        assert trained_model is not None, 'trained model is None'
        if c_type == 'knn':
            # Load test data
            test_data = self.data_handler.load_test()
            test_data = test_data.drop(columns=['timestamp'])
            test_data = test_data.values

            predictions = trained_model.predict(test_data[:, 0:2])
            mse = np.sum((predictions - test_data[:, 2]) ** 2) / len(test_data)

            return mse
        elif c_type == 'other_model':
            pass
        else:
            raise ValueError('Invalid type of classification model')
