#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

import itertools
import numpy as np
import re
from collections import Counter, defaultdict
import tensorflow as tf
import keras

USERS = {"thomas": 0, "matthieu": 0, "fred": 0, "pedro": 0, "romain": 1, "dim": 1,
         "samy": 0, "laurent": 1, "ben": 0, "guillaume": 0
         }


def preprocess_sentence(sentence, w_len):
    """
    Transform sentence so it can be processed as a list of words

    :param sentence: sentence to be processed
    :type sentence: string
    :param w_len: words minimum length
    :type w_len: integer
    :return: list of words
    """
    words = sentence.split()
    words = [w.lower() for w in words if len(w) > w_len]

    #  Keep only letters and numbers
    #words = [re.sub('[^a-z0-9]+', '', w) for w in words]

    return words


def index_sentence(sentence, global_dict):
    """
    :param sentence:
    :param global_dict:
    :return:
    """
    return [global_dict.index(word) for word in sentence if word in global_dict]


def build_words_dict(text_file, w_len=1):
    """

    :param text_file:
    :param w_len:
    :return:
    """
    data_dict = defaultdict(list)
    words_dict = []

    err = 0
    with open(text_file, "r") as data_file:
        for line in data_file:
            try:
                line = line.strip()
                user, sentence = line.split(':')[0], line.split(':')[1]
                sentence = preprocess_sentence(sentence, w_len)
                if sentence:
                    data_dict[user.strip()].append(sentence)
                    words_dict.extend(sentence)
            except Exception as exc:
                err += 1

    # Global dictionary of words used
    return data_dict, words_dict


def build_model(data_dict, words_dict, model_file):
    """

    :param words_dict:
    :return:
    """
    data_processed = defaultdict(list)
    for user, l_sentences in data_dict.items():
        for sentence in l_sentences:
            idx_sentence = index_sentence(sentence, words_dict)
            data_processed[user].append(idx_sentence)

    train_data = []
    train_labels = []
    #for user, l_data in data_processed.items():
    #    for username in USERS.keys():
    #        print(username)
    #        print(user)
    #        if username in user.lower():
    #            label = USERS[username]
    #            print('label: %s' % label)
    #            break
    #    for data in l_data:
    #        train_data.append(data)
    #        train_labels.append(label)
    idx = 0
    for user, l_data in data_processed.items():
        idx += 1
        train_data.append(l_data)
        train_labels.append(user)


    train_data = [keras.preprocessing.sequence.pad_sequences(train_data_l,
                                                            value=0,
                                                            padding='post',
                                                            maxlen=128) for train_data_l in train_data]
    train_data = np.array(train_data)
    # input shape is the vocabulary count used
    vocab_size = len(words_dict)

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(10, activation=tf.nn.sigmoid))

    model.summary()

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_data,
                        train_labels,
                        epochs=500,
                        batch_size=512,
                      # validation_data=(train_data, train_labels),
                        verbose=1)

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_file, "w") as json_file:
        json_file.write(model_json)

    return model


def load_model(model_file):
    """

    :param model_file:
    :return:
    """

    # load json and create model
    with open(model_file, 'r') as json_file:
        loaded_model_json = json_file.read()
        loaded_model = keras.models.model_from_json(loaded_model_json)

    return loaded_model


def ranking_words(data_dict, words_to_rank):
    """

    :param data_dict:
    :param words_to_rank:
    :return:
    """
    ranking = defaultdict(list)
    for user, l_words in data_dict.items():
        words = list(itertools.chain.from_iterable(l_words))
        for word in words_to_rank:
            count = words.count(word)
            if count:
                ranking[word].append((user, words.count(word)))

    return ranking


if __name__ == '__main__':
    import pprint
    import sys

    #data_dict, words_dict = build_words_dict("./conv_clean.txt", 2)

    #d = ranking_words(data_dict, ["evg"])

    #pprint.pprint(d)
    #sys.exit()

    model_file = "/tmp/model10.json"
    text_file = "/home/mvelay/workspace/ML/conv_clean.txt"

    data_dict, words_dict = build_words_dict(text_file)
    build_model = build_model(data_dict, words_dict, model_file)

    model = load_model(model_file)

    while True:
        new_phrase = input("Sentence to analyze:\n")
        sentence = index_sentence(new_phrase.split(), words_dict)
        predictions = model.predict_classes(sentence)
        predictions.flatten()

    #p_result = sum(predictions) / len(predictions)

    #result = ("DROITE", p_result[0] * 100) if p_result > 0.5 else ("GAUCHE", (1 - p_result[0]) * 100)

    #print("****************************")
    #print("Sentence to analyze:\n %s" % new_phrase)
    #print("Result: mec de %s (%s %%)" % (result[0], result[1]))
