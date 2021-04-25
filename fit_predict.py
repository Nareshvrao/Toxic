from toxic.model import get_model, get_model_pool, get_model_deepmoji_style, get_model_pool_gru_cnn
from toxic.nltk_utils import tokenize_sentences
from toxic.train_utils import train_folds
from toxic.embedding_utils import read_embedding_list, clear_embedding_list, convert_tokens_to_ids, \
    read_embedding_list_glove
from toxic.att_lstm_model import get_model_att_lstm
from toxic.gru_avg_max_model import get_gru_model
from toxic.dpcnn import get_dpcnn_model
from toxic.capsnet import get_capsnet_model

import argparse
import numpy as np
import os
import pandas as pd
import time
import pickle

UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

PROBABILITIES_NORMALIZE_COEFFICIENT = 1.4


def main():
    parser = argparse.ArgumentParser(
        description="Recurrent neural network for identifying and classifying toxic online comments")

    parser.add_argument("train_file_path")
    parser.add_argument("test_file_path")
    parser.add_argument("embedding_path")
    parser.add_argument("--result-path", default="toxic_results")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sentences-length", type=int, default=500)
    parser.add_argument("--recurrent-units", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--dense-size", type=int, default=32)
    parser.add_argument("--fold-count", type=int, default=10)
    parser.add_argument("--modelname-prefix", type=str, default="")
    parser.add_argument("--cv", type=str, default="True")
    parser.add_argument("--use-roc", type=str, default="False")

    args = parser.parse_args()

    if args.fold_count <= 1:
        raise ValueError("fold-count should be more than 1")
    print('Input params')
    print(args)

    start = time.time()
    print('#' * 50)
    print("Loading data...")
    print('#' * 50)

    if os.path.exists(os.path.join(args.result_path, 'tokenized_sentences_train.pkl')) and os.path.exists(os.path.join(args.result_path, 'tokenized_sentences_train.pkl')) and os.path.exists(os.path.join(args.result_path, 'tokenized_sentences_train.pkl')):
        print('Preprocessed files found. Reading preprocess files')
        train_data = pd.read_csv(args.train_file_path)
        test_data = pd.read_csv(args.test_file_path)
        y_train = train_data[CLASSES].values

        with open(os.path.join(args.result_path, 'tokenized_sentences_train.pkl'), 'rb') as f:
            tokenized_sentences_train = pickle.load(f)
        with open(os.path.join(args.result_path, 'tokenized_sentences_test.pkl'), 'rb') as f:
            tokenized_sentences_test = pickle.load(f)
        with open(os.path.join(args.result_path, 'words_dict.pkl'), 'rb') as f:
            words_dict = pickle.load(f)

    else:
        print('Preprocessed files not found.')
        train_data = pd.read_csv(args.train_file_path)
        test_data = pd.read_csv(args.test_file_path)

        list_sentences_train = train_data["comment_text"].fillna(NAN_WORD).values
        list_sentences_test = test_data["comment_text"].fillna(NAN_WORD).values
        y_train = train_data[CLASSES].values

        print('#' * 50)
        print("Tokenizing sentences in train set...")
        print('#' * 50)
        tokenized_sentences_train, words_dict = tokenize_sentences(list_sentences_train, {})

        print('#' * 50)
        print("Tokenizing sentences in test set...")
        print('#' * 50)
        tokenized_sentences_test, words_dict = tokenize_sentences(list_sentences_test, words_dict)

        print('Saving preprocess files...')
        with open(os.path.join(args.result_path, 'tokenized_sentences_train.pkl'), 'wb') as f:
            pickle.dump(tokenized_sentences_train, f)
        with open(os.path.join(args.result_path, 'tokenized_sentences_test.pkl'), 'wb') as f:
            pickle.dump(tokenized_sentences_test, f)
        with open(os.path.join(args.result_path, 'words_dict.pkl'), 'wb') as f:
            pickle.dump(words_dict, f)

    print('total words', len(words_dict))
    words_dict[UNKNOWN_WORD] = len(words_dict)

    print('#' * 50)
    print("Loading embeddings...")
    print('#' * 50)
    if 'glove' in args.embedding_path:
        print('Reading Glove embedding')
        embedding_list, embedding_word_dict = read_embedding_list_glove(args.embedding_path)
    else:
        print('Reading Fasttext embedding')
        embedding_list, embedding_word_dict = read_embedding_list(args.embedding_path)

    embedding_size = len(embedding_list[0])
    print('Embedding size', embedding_size)

    print('#' * 50)
    print("Preparing data...")
    print('#' * 50)
    embedding_list, embedding_word_dict = clear_embedding_list(embedding_list, embedding_word_dict, words_dict)

    embedding_word_dict[UNKNOWN_WORD] = len(embedding_word_dict)
    embedding_list.append([0.] * embedding_size)
    embedding_word_dict[END_WORD] = len(embedding_word_dict)
    embedding_list.append([-1.] * embedding_size)

    embedding_matrix = np.array(embedding_list)
    print('Embedding matrix shape:', embedding_matrix.shape)

    id_to_word = dict((id, word) for word, id in words_dict.items())
    train_list_of_token_ids = convert_tokens_to_ids(
        tokenized_sentences_train,
        id_to_word,
        embedding_word_dict,
        args.sentences_length)
    test_list_of_token_ids = convert_tokens_to_ids(
        tokenized_sentences_test,
        id_to_word,
        embedding_word_dict,
        args.sentences_length)
    X_train = np.array(train_list_of_token_ids)
    X_test = np.array(test_list_of_token_ids)

    # GRU cross validation
    # get_model_func = lambda: get_model(
    #     embedding_matrix,
    #     args.sentences_length,
    #     args.dropout_rate,
    #     args.recurrent_units,
    #     args.dense_size)

    # GRU maxpool, avgpool
    # get_model_func = lambda: get_model_pool(
    #     embedding_matrix,
    #     args.sentences_length,
    #     args.dropout_rate,
    #     args.recurrent_units,
    #     args.dense_size)

    # deepmoji style
    # get_model_func = lambda: get_model_deepmoji_style(
    #     embedding_matrix,
    #     args.sentences_length,
    #     args.dropout_rate,
    #     args.recurrent_units,
    #     args.dense_size)

    # GRU maxpool, avgpool validation
    # get_model_func = lambda: get_gru_model(
    #     embedding_matrix,
    #     args.sentences_length,
    #     args.dropout_rate,
    #     args.recurrent_units,
    #     args.dense_size)

    # GRU maxpool, avgpool + cnn validation
    # get_model_func = lambda: get_model_pool_gru_cnn(
    #     embedding_matrix,
    #     args.sentences_length,
    #     args.dropout_rate,
    #     args.recurrent_units,
    #     args.dense_size)

    # dpcnn validation
    get_model_func = lambda: get_dpcnn_model(
        embedding_matrix,
        args.sentences_length,
        args.dropout_rate,
        args.dense_size)

    # lstm with attention cross val
    # get_model_func = lambda: get_model_att_lstm(
    #     embedding_matrix,
    #     args.sentences_length,
    #     args.dropout_rate,
    #     args.recurrent_units,
    #     args.dense_size)

    # capsule net
    # get_model_func = lambda: get_capsnet_model(
    #     embedding_matrix,
    #     args.sentences_length,
    #     args.dropout_rate,
    #     args.recurrent_units,
    #     args.dense_size)

    print('#' * 50)
    print("Starting to train models...")
    print('#' * 50)
    models = train_folds(X_train, y_train, args.fold_count, args.batch_size, get_model_func, args.cv, args.use_roc)

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    print('#' * 50)
    print("Predicting results...")
    print('#' * 50)

    if args.cv == "True":
        test_predicts_list = []
        for fold_id, model in enumerate(models):
            model_path = os.path.join(args.result_path,
                                      "{0}_model{1}_weights.npy".format(args.modelname_prefix, fold_id))
            np.save(model_path, model.get_weights())

            test_predicts_path = os.path.join(args.result_path,
                                              "{0}_test_predicts{1}.npy".format(args.modelname_prefix, fold_id))
            test_predicts = model.predict(X_test, batch_size=args.batch_size * 2)
            test_predicts_list.append(test_predicts)
            np.save(test_predicts_path, test_predicts)

        test_predicts = np.ones(test_predicts_list[0].shape)
        for fold_predict in test_predicts_list:
            test_predicts *= fold_predict

        test_predicts **= (1. / len(test_predicts_list))
        # test_predicts **= PROBABILITIES_NORMALIZE_COEFFICIENT

        test_ids = test_data["id"].values
        test_ids = test_ids.reshape((len(test_ids), 1))

        test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
        test_predicts["id"] = test_ids
        test_predicts = test_predicts[["id"] + CLASSES]
        submit_path = os.path.join(args.result_path, "{}_submit".format(args.modelname_prefix))
        test_predicts.to_csv(submit_path, index=False)
        print('#' * 50)
        print('Prediction Completed...')
        print('#' * 50)
        total_time = time.time() - start
        mins, sec = divmod(total_time, 60)
        hrs, mins = divmod(mins, 60)
        print('Total time taken : {:.0f}h {:.0f}m {:.0f}s'.format(hrs, mins, sec))
    else:
        print('No Cross Validation')
        test_predicts = models.predict(X_test, batch_size=args.batch_size * 2)
        test_ids = test_data["id"].values
        test_ids = test_ids.reshape((len(test_ids), 1))
        test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
        test_predicts["id"] = test_ids
        test_predicts = test_predicts[["id"] + CLASSES]
        submit_path = os.path.join(args.result_path, "{}_submit_nocv".format(args.modelname_prefix))
        test_predicts.to_csv(submit_path, index=False)

        total_time = time.time() - start
        mins, sec = divmod(total_time, 60)
        hrs, mins = divmod(mins, 60)
        print('Total time taken : {:.0f}h {:.0f}m {:.0f}s'.format(hrs, mins, sec))


if __name__ == "__main__":
    main()
