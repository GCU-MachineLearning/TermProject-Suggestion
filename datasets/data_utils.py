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



""" << 지울 주석 >>
우선 데이터 로드를 ratings, user, item 그리고 test 위한 test 정도로 두었 습니다. 
참조 자료는 ( https://velog.io/@kms1003/추천시스템-01.-기본적인-추천시스템 ) 입니다. 
혹 더 필요 하시 다면, 아래의 load_~ 함수를 추가 해 주시면 될 것 같습니다. 

"""
print("Dd")


class Data(object):
    def __init__(self, data_dir, dataset):
        print("1")
        """
        :param: data_dir: directory of
        :param: dataset: dataset name
        """
        self.data_dir = data_dir
        self.dataset = dataset

    def load_data(self):
        print("2")
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
        print("3")
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

    """
    # TODO: Data preprocessing 
    # 여기에 데이터 전처리 코드를 작성해 주시면 됩니다. 
    # 이후, main 틀 에서 data_handler.preprocess()를 호출 하여 진행 합니다. 
    
    # 혹, preprocessing 을 진행 하기 위한 helper function 이 필요 하다면,
    # 아래와 같이 _function 을 사용 하여 이 class 내부 에서만 호출 하도록 진행 해 주세요.
    
    # 그런데 데이터 가 너무 깨끗 해서, 어느 부분에 대해 전처리 를 진행 해야 할 지 잘 모르겠습니다. 
    """
    def _helper_function(self):
        print("4")
        """
        :param: 추가 하실 파라미터 에 대한 간략한 설명
        :param: 또 다른 파라미터 에 대한 간략한 설명
        :return: return 값에 대한 간략한 설명
        """
        pass  # helper function for preprocessing.

    def preprocess(self):
        print("5")
        """
        :param: 추가 하실 파라미터 에 대한 간략한 설명
        :return: return 값에 대한 간략한 설명
        """
        # 아래는 예제 코드로, 사용자 에 대한 데이터(나이, 성별, 직업, 지역) / 영화에 대한 데이터(장르) / 평점 데이터 를 로드할 수 있습니다.
        ratings, n_users, n_items = self.load_data()
        user_frame = self.load_user()
        item_frame = self.load_item()
        test_data = self.load_test()
        self._helper_function()
        
        # 의미없고 null 값이 대부분인 칼럼들을 item_frame에서 drop
        item_frame.drop(item_frame.columns[3], axis =1, inplace=True)    
        item_frame.drop(item_frame.columns[5], axis =1, inplace=True)    
        item_frame.dropna(axis=0, inplace=True)
        
        # 3가지 데이터프레임에서의 null 값 확인
        print(user_frame.isnull().sum())    
        print(item_frame.isnull().sum())    
        print(ratings.isnull().sum())
        
        # user_frame에서 성별,직업 칼럼을 떼어내서 인코딩
        le = LabelEncoder()    
        e1 = pd.DataFrame(le.fit_transform(user_frame['sex']), columns = ['sex'])
        e2 = pd.DataFrame(le.fit_transform(user_frame['occupation']), columns = ['occupation'])

        # user_frame에서 성별, 직업 칼럼 drop
        user_frame = user_frame.drop(['sex', 'occupation'], axis=1)

        # user_frame에 인코딩한 새로운 두 칼럼 concat
        user_frame = pd.concat([user_frame, e1], axis=1)
        user_frame = pd.concat([user_frame, e2], axis=1)
        
        return ratings, n_users, n_items, user_frame, item_frame, test_data

