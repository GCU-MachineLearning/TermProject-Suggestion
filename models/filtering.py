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
        """
        :param: data_handler: data handler
        """
        self.data_handler = data_handler

    def model_based(self):
        """
        Model-based collaborative filtering
        Library used: ?
        """
        pass

    def memory_based(self):
        """
        Memory-based collaborative filtering
        """
        pass

    def hybrid(self):
        """
        Hybrid collaborative filtering
        """
        pass
