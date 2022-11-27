import datasets.data_utils as data_utils
from models import ml, filtering


def main(root_dir, dataset):

    # DECLARE HANDLER
    # - DATA HANDLER: data_handler.load_data() / load_user() / load_item() / load_test()
    # - ML HANDLER: ml_handler.train() / test()
    # - FILTERING HANDLER: filtering_handler.train() / test()
    data_handler = data_utils.Data(root_dir, dataset)
    ratings, n_users, n_items, user_frame, item_frame, test_data = data_handler.preprocess()
    print(f'<<Load data>> - Successfully loaded clean data')

    # Throw data_handler to below model handler
    # so that can handle data in model handler if needed
    ml_handler = ml.ML(data_handler)
    filtering_handler = filtering.Filtering(data_handler)

    # Step 1: train model using preprocessed data
    """
    # P, Q = latent vector
    # b_u, b_i = each bias (user, item)
    # b = global bias
    """

    is_pretrained = True
    save = False
    iteration = 10000

    if is_pretrained:
        print("<<Load pretrained model>> - Matrix Factorisation, iteration: {}".format(iteration))
        P, Q, b_u, b_i, b = ml_handler.load_model(iteration=iteration + 1)
    else:
        print("\n<<Training>> - Matrix Factorisation", end='\n')
        P, Q, b_u, b_i, b, _ = ml_handler.matrix_factorization(steps=iteration + 1, save=save)

    # Step 3: test model using test data
    mf_mse = ml_handler.test_matrix_factorisation(P, Q, b_u, b_i, b)

    print("\n<<Compare>> - Matrix Factorisation vs Classification")
    # compare to simple classification model
    classifications = ['knn', 'decisionTree', 'randomForest', 'svm']
    # classifications = ['knn', 'decisionTree', 'randomForest']
    classifications_mse = []

    for classification in classifications:
        print('|')
        print("|\t<<Training>> - Classification: {}".format(classification))
        _trained = ml_handler.classification(classification)
        print("|\t<<Testing>> - Classification: {}".format(classification))
        _mse = ml_handler.test_classification(_trained, classification)
        classifications_mse.append([classification, _mse])

    classifications_mse = sorted(classifications_mse, key=lambda x: x[1])

    print("\n<<Compare>> - Matrix Factorisation vs Classification")
    print(f'Matrix Factorisation MSE: {mf_mse: .4f}')
    for classification in classifications_mse:
        print(f'{classification[0]} MSE: {classification[1]: .3f}', end=' | ')

    # suggest items to user via classificatoin
    user_data = data_handler.load_user()
    user_id = 1
    movie_list = ml_handler.movie_suggestion_mf(user_id, [P, Q, b_u, b_i, b])
    print("\n\n<<Suggestion via Matrix Factorisation>>")
    print(
        f'For user [Age: {user_data["age"][user_id]}, Gender: {user_data["sex"][user_id]}], who works as {user_data["occupation"][user_id]}...')
    for movie in movie_list[:5]:
        print(f'\tMovie title: {movie[0]}, predicted rating: {movie[1]}')

    user_id = 1
    print("\n<<Suggestion via SVD>>")
    print(
        f'For user [Age: {user_data["age"][user_id]}, Gender: {user_data["sex"][user_id]}], who works as {user_data["occupation"][user_id]}...')
    result = filtering_handler.svd(user_id, 5)  # recommend 5 movies
    for movie in result['movie_title']:
        print(f'\tMovie title: {movie}')

    print("\n<<Suggestion via user-based filtering>>")
    print(
        f'For user [Age: {user_data["age"][user_id]}, Gender: {user_data["sex"][user_id]}], who works as {user_data["occupation"][user_id]}...')
    result = filtering_handler.user_based_recommend(user_id)
    for movie in result:
        print(f'\tMovie title: {movie}')

    item_id = 1
    print("\n<<Suggestion via item-based filtering>>")
    result = filtering_handler.item_based_recommend(item_id)
    for movie in result:
        print(f'\tMovie title: {movie}')

    title = 'Toy Story (1995)'
    print("\n<<Suggestion via content-based filtering>>")
    result = filtering_handler.content_based_recommend(title)
    result = result.values.tolist()
    for movie in result:
        if movie == title: continue # print except the input film
        print(f'\tMovie title: {movie}')

    print("\n<<Suggestion finished>>\n")
    

if __name__ == '__main__':
    ROOT_DIR = 'datasets/ml-100k'
    DATASET = ROOT_DIR.split('/')[-1]

    main(ROOT_DIR, DATASET)
