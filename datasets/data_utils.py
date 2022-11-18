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
        item_frame = pd.read_csv(os.path.join(self.data_dir, 'u.item'), sep='|', header=None,
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

    """
    # TODO: Data preprocessing 
    # Please preprocess data in below function. 
    # Then, call this function in main.py, by using data_handler object. 
    
    # You can load various data by self.~~  
    """
    def preprocess(self):
        """
        :param: parameter that you want to add
        :return: preprocessed data
        """
        # Below code is sample codes
        ratings, n_users, n_items = self.load_data()
        user_frame = self.load_user()
        item_frame = self.load_item()
        test_data = self.load_test()
        return ratings, n_users, n_items, user_frame, item_frame, test_data

