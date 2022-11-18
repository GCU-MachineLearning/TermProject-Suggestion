"""
Module for handling filtering algorithms
3 methods are provided:
    1. model-based collaborative filtering: SVD, Matrix Factorization
    2. memory-based filtering: User-based / Item-based
    3. hybrid filtering: (model-based + memory-based)
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
        Library used: sklearn
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