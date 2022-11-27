"""
Module for handling filtering algorithms
3 methods are provided:
    1. model-based collaborative filtering: SVD, (Matrix Factorization << 지울 주석: 이 부분은 ml.py 에 있습니다. >>)
    2. memory-based filtering: User-based / Item-based
    3. hybrid filtering: (model-based + memory-based)
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

    def content_based_recommend(self, title):
        import pandas as pd

        # load ratings data from each user
        df = self.data_handler.load_test()
        # load movie info data
        movie_titles = self.data_handler.load_item()

        # extract two columns and rename them for future merge
        movie_titles = movie_titles[["movie_id", "movie_title"]]
        movie_titles = movie_titles.rename(columns={'movie_id': 'item_id', 'movie_title': 'title'})

        # merge user ratings dataframe and movie info dataframe
        df = pd.merge(df, movie_titles, on='item_id')

        # title + average rating data we viewed above and recreate it in a separate dataframe giving films alongside their average ratings
        ratings_df = pd.DataFrame(df.groupby('title')['rating'].mean())
        ratings_df.rename(columns={'rating': 'average_rating'}, inplace=True)

        # adding number of ratings data from ratings info df
        ratings_df['num_of_ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())

        # convert df into the equivalent of an X matrix
        user_movie_matrix = df.pivot_table(values='rating' , index='user_id' , columns='title' )

        # get user ratings for film
        film_x_user_ratings = user_movie_matrix[title]
        # create pandas series of correlations for all films with film_x
        similar_to_film_x = user_movie_matrix.corrwith(film_x_user_ratings)
        # convert to df
        corr_film_x = pd.DataFrame(similar_to_film_x, columns=['Correlation'])
        # drop nulls
        corr_film_x.dropna(inplace=True)
        # join ratings info to enbale filtering of films with low nums of ratings
        corr_film_x = corr_film_x.join(ratings_df['num_of_ratings'])
        # apply min number of reviews(30) filter
        new_corr_film_x = corr_film_x[corr_film_x['num_of_ratings'] >= 30]
        # apply min correlation(0.9) filter
        new_corr_film_x = corr_film_x[corr_film_x['Correlation'] >= 0.9]
        # sort into ascending order
        new_corr_film_x = new_corr_film_x.sort_values('Correlation',ascending=False).head(20)
        # return films that have correlation larger than 0.9
        result = new_corr_film_x.reset_index()['title']
        return result

