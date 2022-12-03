"""
Module for data loading and preprocessing.
4 method are provided:
    1. load_data: load data from file (user_id / item_id / rating / timestamp)
    2. load_user: load user information from file (age, sex, occupation, zip)
    3. load_item: load item information from file (movie_id / movie_title / release_date / video_release_date / IMDb_URL / ... )

    4. load_test: load test data from file (user_id / item_id / rating / timestamp)
Dataset: ml-100k (http://grouplens.org/datasets/movielens/100k/)
"""

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Data(object):
    def __init__(self, data_dir, dataset):
        """
        :param: data_dir: directory of
        :param: dataset: dataset name
        """
        self.data_dir = data_dir
        self.dataset = dataset

    def load_data(self):
        """
        Load data from data directory.
        :return: dataframe, n_users, n_items
        """
        data = pd.read_csv(os.path.join(self.data_dir, 'u.data'), sep='\t', header=None,
                           names=['user_id', 'item_id', 'rating', 'timestamp'])
        n_users = data['user_id'].unique().shape[0]
        n_items = data['item_id'].unique().shape[0]
        return data, n_users, n_items

    def load_user(self):
        """
        Load user information.
        :return: user_frame
        """
        user_frame = pd.read_csv(os.path.join(self.data_dir, 'u.user'), sep='|', header=None,
                                 names=['age', 'sex', 'occupation', 'zip_code'])
        return user_frame

    def load_item(self):
        """
        Load item information.
        :return: item_frame
        """
        item_frame = pd.read_csv(os.path.join(self.data_dir, 'u.item'), sep='|', header=None, encoding='latin-1',
                                 names=['movie_id', 'movie_title', 'release_date', 'video_release_date',
                                        'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                                        'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                        'Thriller', 'War', 'Western'])
        return item_frame

    def load_test(self):
        """
        Load test data.
        :return: test_data
        """
        test_data = pd.read_csv(os.path.join(self.data_dir, 'ua.test'), sep='\t', header=None,
                                names=['user_id', 'item_id', 'rating', 'timestamp'])
        return test_data


    def preprocess(self):
        """
        :param: NaN
        :return: all processed data
        """
        ratings, n_users, n_items = self.load_data()
        user_frame = self.load_user()
        item_frame = self.load_item()
        test_data = self.load_test()

        # drop rows which have missing/NaN values
        item_frame.drop(item_frame.columns[3], axis=1, inplace=True)
        item_frame.drop(item_frame.columns[5], axis=1, inplace=True)
        item_frame.dropna(axis=0, inplace=True)

        # check NaN values from below three dataframes.
        # If there are NaN values, assert will be raised.
        assert ratings.isnull().values.any() == False, "NaN values exist in ratings"
        assert user_frame.isnull().values.any() == False, "NaN values exist in user_frame"
        assert item_frame.isnull().values.any() == False, "NaN values exist in item_frame"

        # encode categorical data
        le = LabelEncoder()
        e1 = pd.DataFrame(le.fit_transform(user_frame['sex']), columns=['sex'])
        e2 = pd.DataFrame(le.fit_transform(user_frame['occupation']), columns=['occupation'])
        e1.index=e1.index + 1
        e2.index=e2.index + 1       

        # drop original columns, ['sex', 'occupation']
        user_frame = user_frame.drop(['sex', 'occupation'], axis=1)

        # concat encoded columns
        user_frame = pd.concat([user_frame, e1], axis=1)
        user_frame = pd.concat([user_frame, e2], axis=1)

        # return processed data
        return ratings, n_users, n_items, user_frame, item_frame, test_data
