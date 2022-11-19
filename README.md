# Machine Learning Term Project - Suggestion program 

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