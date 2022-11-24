import datasets.data_utils as data_utils
from models import ml, filtering


""" << 지울 주석 >> 
Main 에서 위에서 정의한 handler 들을 사용 하여 
 - data load + data preprocessing - data_handler
 - 다양한 ML model 들 (matrix factorisation 및 clustering / classification) - ml_handler
 - 다양한 filtering model 들 (memory-based / model-based / hybrid) - filtering_handler
을 진행 하게 됩니다. 

<< TODO >>
data preprocessing 의 경우 ./datasets/data_utils.py 의 Data class      내부의 preprocess() 함수를 작성 해 주시면 됩니다. 
machine learning   의 경우 ./models/ml.py           의 ML class        내부의 matrix_factorisation, classification, clustering 함수를 작성 해 주시면 됩니다.
filtering          의 경우 ./models/filtering.py    의 Filtering class 내부의 model_based, memory_based, hybrid 함수를 작성 해 주시면 됩니다.

이 코드를 colab 등에서 실행 시키 시려면, 
1. 이 코드를 google drive 에 올리고,
2. google drive 에서 colab 을 실행 시키고, colab 에서 google drive 에 접근한 뒤, main.py 가 존재하는 경로로 cd 한 뒤,
3. !python main.py 커맨드 를 입력 하면 됩니다. 혹은 !sh runner.sh (일반적인 터미널 명령 앞에 ! 붙여 주면 colab 에서 실행 가능 합니다.)
"""


def main(root_dir, dataset):

    # DECLARE HANDLER
    # - DATA HANDLER: data_handler.load_data() / load_user() / load_item() / load_test()
    # - ML HANDLER: ml_handler.train() / test()
    # - FILTERING HANDLER: filtering_handler.train() / test()
    data_handler = data_utils.Data(root_dir, dataset)
    ratings, n_users, n_items, user_frame, item_frame, test_data = data_handler.preprocess()

    # Throw data_handler to below model handler
    # so that can handle data in model handler if needed
    ml_handler = ml.ML(data_handler)
    filtering_handler = filtering.Filtering(data_handler)

    """ << 지울 주석 >> 
    위의 과정 까지 handler 들을 선언 하고, 실제 pipeline 은 아래 부터 구성 됩니다. 
    """

    # Step 1: train model using preprocessed data
    """
    # P, Q = 각 user, item 의 latent vector
    # b_u, b_i = 각 user, item 의 bias
    # b = global bias
    # training_process = 매 step 마다의 loss 값: 혹시 몰라서 저장 하도록 했는데, 필요 없으면 버려 주세요. 
    """

    print("\n<<Training>> - Matrix Factorisation")
    P, Q, b_u, b_i, b, _ = ml_handler.matrix_factorization()

    # Step 3: test model using test data
    mf_mse = ml_handler.test_matrix_factorisation(P, Q, b_u, b_i, b)

    print("\n<<Compare>> - Matrix Factorisation vs Classification")
    # compare to simple classification model
    c_model = 'knn'
    knn = ml_handler.classification(c_model)
    knn_mse = ml_handler.test_classification(knn, c_model)

    print(f'MF MSE: {mf_mse:.4f}, KNN MSE: {knn_mse:.4f}')
    print(f'MF is better than KNN: {mf_mse < knn_mse}')

    # suggest items to user
    user_data = data_handler.load_user()
    user_id = 1
    movie_list = ml_handler.movie_suggestion_mf(user_id, [P, Q, b_u, b_i, b])
    print("\n<<Suggestion>>")
    print(f'For user [Age: {user_data["age"][user_id]}, Gender: {user_data["sex"][user_id]}], who works as {user_data["occupation"][user_id]}...')
    for movie in movie_list[:5]:
        print(movie)


if __name__ == '__main__':
    ROOT_DIR = 'datasets/ml-100k'
    DATASET = ROOT_DIR.split('/')[-1]

    main(ROOT_DIR, DATASET)
