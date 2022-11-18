import datasets.data_utils as data_utils
from models import ml, filtering


def main(root_dir, dataset):

    # DECLARE HANDLER
    # - DATA HANDLER: data_handler.load_data() / load_user() / load_item() / load_test()
    # - ML HANDLER: ml_handler.train() / test()
    # - FILTERING HANDLER: filtering_handler.train() / test()
    data_handler = data_utils.Data(root_dir, dataset)

    # Throw data_handler to below model handler
    # so that can handle data in model handler if needed
    ml_handler = ml.ML(data_handler)
    filtering_handler = filtering.Filtering(data_handler)

    # Step 1: preprocess data using handler
    preprocessed_data = data_handler.preprocess()

    # Step 2: train model using preprocessed data
    ml_handler.train(preprocessed_data)
    filtering_handler.train(preprocessed_data)


if __name__ == '__main__':
    ROOT_DIR = 'datasets/ml-100k'
    DATASET = ROOT_DIR.split('/')[-1]

    main(ROOT_DIR, DATASET)
