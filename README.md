# Toxic-word

Classify the toxic word in the sentence. 

Dataset : https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

# Tools required
Python 3.5+
NLTK
Numpy
Pandas
Sklearn
Keras
Tensorflow
Glove word embedding
Fasttext word embedding

# How to run the Notebook

Host in google colab or jupyter

# How to run the program
Model needs to be set in the fit_predict.py. The code trains the model with k fold cross validation. It save the trained model and predictions.

python fit_predict.py train_data_path test_data_path pretrained_embedding_path --result-path --sentences-length --fold-count --dense-size --modelname-prefix --batch-size --dropout-rate

Sample execution command

python fit_predict.py ./data/train.csv ./data/test.csv ./NLP/fasttext_embedding/crawl-300d-2M.vec --result-path ./toxic_results --sentences-length 400 --fold-count 10 --dense-size 256 --modelname-prefix dpcnn400_fasttextcrawl --batch-size 512 --dropout-rate 0.4

