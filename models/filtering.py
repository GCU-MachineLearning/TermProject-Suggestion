"""
Module for handling filtering algorithms
3 methods are provided:
    1. model-based collaborative filtering: SVD, (Matrix Factorization << 지울 주석: 이 부분은 ml.py 에 있습니다. >>)
    2. memory-based filtering: User-based / Item-based
    3. hybrid filtering: (model-based + memory-based)
"""

"""
<< 지울 주석 >>
Proposal 상에서 
Matrix Factorization / Clustering / Classification 은 ML 에서 진행 하고,
User-based / Item-based / Hybrid Filtering 은 Filtering 에서 진행 하도록 되어 있습니다.

때문에 Matrix factorization 은 ml.py 에서 구현 되도록 진행 되었습니다. 
ml.py 에서 구현 된 Matrix factorization 은 user_id, item_id 를 통해 ratings 를 예측하는 방식으로 진행되고 있습니다. 

(N 이라는 사용자에 대해서, X 라는 영화에 대한 평점을 예측함으로써, N에게 X 라는 영화가 추천(= 높은 평점) 할지, 아닐지(= 낮은 평점)를 판단합니다.)

때문에 여기서는 위 방법이 아닌, data_handler 에서 user 와 item 의 정보를 통해 user-based 혹은 item-based filtering 을 진행 해도 될 것 같습니다.
"""


class Filtering:
    def __init__(self, data_handler):
        self.data_handler = data_handler

    def svd(self, userID, num_recommendations):
        import numpy as np
        import pandas as pd
        from scipy.sparse.linalg import svds

        ratings = self.data_handler.load_test()

        # make the format of ratings matrix to be one row per user and one column per movie
        Ratings = ratings.pivot(
            index='user_id', columns='item_id', values='rating').fillna(0)
        R = Ratings.values
        user_ratings_mean = np.mean(R, axis=1)
        Ratings_demeaned = R - user_ratings_mean.reshape(-1, 1)

        U, sigma, Vt = svds(Ratings_demeaned, k=50)

        # to leverage matrix multiplication to get predictions, convert the Σ (now are values) to the diagonal matrix form
        sigma = np.diag(sigma)

        # add the user means back to get the actual star ratings prediction
        all_user_predicted_ratings = np.dot(
            np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

        preds = pd.DataFrame(all_user_predicted_ratings,
                             columns=Ratings.columns)
        movies = self.data_handler.load_item()
        movies = movies[["movie_id", "movie_title"]]
        movies = movies.rename(columns={'movie_id': 'item_id'})

        ratings = self.data_handler.load_test()

        # Get and sort the user's predictions
        user_row_number = userID - 1  # User ID starts at 1, not 0
        sorted_user_predictions = preds.iloc[user_row_number].sort_values(
            ascending=False)  # User ID starts at 1

        # Get the user's data and merge in the movie information.
        user_data = ratings[ratings.user_id == (userID)]
        user_full = (user_data.merge(movies, how='left', left_on='item_id', right_on='item_id').
                     sort_values(['rating'], ascending=False)
                     )

        print('User {0} has already rated {1} movies.'.format(
            userID, user_full.shape[0]))
        print('Recommending highest {0} predicted ratings movies not already rated.'.format(
            num_recommendations))

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations = (movies[~movies['item_id'].isin(user_full['item_id'])].
                           merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                                 left_on='item_id',
                                 right_on='item_id').
                           rename(columns={user_row_number: 'Predictions'}).
                           sort_values('Predictions', ascending=False).
                           iloc[:num_recommendations, :-1]
                           )

        # return recommendations - top similar users rated movies
        return recommendations

    def user_based_recommend(self, user_id, k=20, top_n=5):
        import pandas as pd

        test_df = self.data_handler.load_test()

        # create user-tiem matrix where the rows will be the users, the columns will be the movies
        # and the datafrane us filled with the rating the user has given.
        user_item_m = pd.pivot_table(
            test_df, values='rating', index='user_id', columns='item_id').fillna(0)

        # Similarity between vectors, we want to find a proximity measure between all users using the cosine similarity.
        from sklearn.metrics.pairwise import cosine_similarity
        X_user = cosine_similarity(user_item_m)

        # Get location of the actual movie in the User-Items matrix
        user_ix = user_item_m.index.get_loc(user_id)
        # Use it to index the User similarity matrix
        user_similarities = X_user[user_ix]
        # obtain the indices of the top k most similar users
        most_similar_users = user_item_m.index[user_similarities.argpartition(
            -k)[-k:]]
        # Obtain the mean ratings of those users for all movies
        rec_movies = user_item_m.loc[most_similar_users].mean(
            0).sort_values(ascending=False)
        # Discard already seen movies
        m_seen_movies = user_item_m.loc[user_id].gt(0)
        seen_movies = m_seen_movies.index[m_seen_movies].tolist()
        rec_movies = rec_movies.drop(seen_movies).head(top_n)
        rec_movies = rec_movies.index.to_frame().reset_index(drop=True)

        movies = self.data_handler.load_item()
        movies = movies[["movie_id", "movie_title"]]

        movie_lists = rec_movies.values.tolist()

        result = []
        # change id values to corresponding movie title
        for i in range(0, len(movie_lists)):
            for j in range(0, len(movies["movie_id"].axes[0])):
                if movie_lists[i] == movies["movie_id"][j]:
                    result.append(movies["movie_title"][j])
                    break

        # return recommendations - top similar users rated movies
        return result

    def item_based_recommend(self, item_id, k=5):
        import pandas as pd

        test_df = self.data_handler.load_test()

        # create user-tiem matrix where the rows will be the users, the columns will be the movies
        # and the datafrane us filled with the rating the user has given.
        user_item_m = pd.pivot_table(
            test_df, values='rating', index='user_id', columns='item_id').fillna(0)

        # Similarity between vectors, we want to find a proximity measure between all movies using the cosine similarity.
        from sklearn.metrics.pairwise import cosine_similarity
        X_item = cosine_similarity(user_item_m.T)
        items = self.data_handler.load_item()

        liked = items.loc[items.movie_id.eq(item_id), 'movie_title'].item()
        print(f"\tBecause you liked <{liked}>, we'd recommend you to watch...")
        # get index of movie
        ix = user_item_m.columns.get_loc(item_id)
        # Use it to index the Item similarity matrix
        i_sim = X_item[ix]
        # obtain the indices of the top k most similar items
        most_similar_itmes = user_item_m.columns[i_sim.argpartition(
            -(k+1))[-(k+1):]]

        movies = self.data_handler.load_item()
        movies = movies[["movie_id", "movie_title"]]

        movie_lists = most_similar_itmes.values.tolist()

        result = []
        # change id values to corresponding movie title
        for i in range(0, len(movie_lists)):
            for j in range(0, len(movies["movie_id"].axes[0])):
                if movie_lists[i] == movies["movie_id"][j]:
                    result.append(movies["movie_title"][j])
                    break

        # return recommendations - top similar users rated movies
        return result

    def hybrid(self):
        """
        Hybrid collaborative filtering
        """
        pass
