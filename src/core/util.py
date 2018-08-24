#!/usr/bin/python
# -*- coding: utf-8 -*-
# contains utility functions, also takes care of creation of vocabulary and word embedding files

import numpy as np
import tensorflow as tf
import re
import os
import time

from tensorflow.contrib.tensorboard.plugins import projector
from distutils.dir_util import copy_tree
from urllib.request import urlretrieve
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import zipfile
import _pickle as pickle

vocab = {}
vectors = {}

# logging of question types, but no further usage in this work
question_types = {
    "what": 0,
    "who": 1,
    "why": 2,
    "how": 3,
    "where": 4
}


def get_question_keys():
    keys = ["other", "what", "who", "why", "how", "where"]
    return keys


# first entry of vocab reserved for empty word (zero vector for padding)
def init_vocab(dim):
    vocab[''] = 0
    vectors[0] = np.zeros(dim)


def restore_vocab(embedding_dir):
    if (os.path.exists(embedding_dir + "/vocab.pickle") and os.path.exists(
            embedding_dir + "/vectors.pickle")):
        global vocab, vectors
        vectors_array, rev_vocab = load_embeddings(embedding_dir)
        vectors = {k: v for k, v in enumerate(vectors_array)}
        vocab = dict(zip(rev_vocab.values(), rev_vocab.keys()))


# load embeddings from files
def load_embeddings(embedding_dir):
    print("Loading vectors.pickle and vocab.pickle from %s" % embedding_dir)
    with open(embedding_dir + "/vectors.pickle", 'rb') as handle:
        loaded_vectors = pickle.load(handle)
    with open(embedding_dir + "/vocab.pickle", 'rb') as handle2:
        loaded_vocab = pickle.load(handle2)
    print("Loaded vocab of length %d" % len(loaded_vocab))
    return loaded_vectors, loaded_vocab

# load glove model into dictionary
def loadGloveModel(gloveFile):
    print("Loading pretrained GloVe embeddings from %s" % gloveFile)
    word2vec = {}
    fin = open(gloveFile, encoding="utf8")
    for i, line in enumerate(fin):
        items = line.replace('\r', '').replace('\n', '').split(' ')
        if len(items) < 10:
            continue
        word = items[0]
        vect = np.array([float(i) for i in items[1:] if len(i) >= 1])
        word2vec[word] = vect.tolist()

        if i % 10000 == 0:
            print("Loaded %d vectors already" %i)

    return word2vec


# remove special characters and lowercase words for finding them in GloVe
def normalize_text(s):
    special_chars = ',`´&.:!?;()$\"\''
    norm = ''.join(re.findall('[^' + special_chars + ']', s)).strip().lower()
    norm = list(filter(None, norm.split()))
    # print(norm)
    return norm


def get_word_vector(embeddings, word, size, warn_no_embeddings=False):
    """
    gets index of the word in the stored vocabulary, or updates the vocabulary if the word is not stored yet
    :param embeddings:
    :param word:
    :param size:
    :param warn_no_embeddings: prints warning if word is not in vocabulary yet and no pretrained embeddings are provided
    :return:
    """
    # print("vocab with %d entries" % len(vocab))
    if word in vocab:
        index = vocab[word]
    else:
        index = len(vocab)

        if not embeddings and warn_no_embeddings:
            print("New word %s in vocab recognized, please provide pretrained embeddings to look for this word's vector"
                  % word)
            return False
        elif word in embeddings:
            vec = embeddings[word]
        # TODO unknown words during evaluation are each assigned their own random vector
        # TODO could implement a vocabulary entry for unknown words
        else:
            vec = np.random.uniform(-1.3, 1.3, size)
        vocab[word] = index
        vectors[index] = vec
    return index


# save vocab end word embedding representations
def save_embeddings(embedding_dir, dim):
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)
    embedding = np.empty((len(vocab), dim), dtype=np.float32)
    for key, value in vocab.items():
        embedding[value] = vectors[value]

    rev_vocab = dict(zip(vocab.values(), vocab.keys()))

    with open(embedding_dir + '/vocab.pickle', 'wb') as handle:
        pickle.dump(rev_vocab, handle)
    with open(embedding_dir + '/vectors.pickle', 'wb') as handle:
        pickle.dump(embedding, handle)

    print("Saved embeddings to %s with vocab size %d" % (embedding_dir, len(vocab)))
    return len(rev_vocab)

# create backup copy of model after each training epoch
def copy_model(src, gs):
    dest = src + "_" + str(gs)
    copy_tree(src, dest)


# save validation / testing results to score file
def save_eval_score(entry):
    with open('outputs/scores.txt', 'a') as f:
        f.write(entry + '\n')


# save a copy of the config file's settings to the model folder to save the used hyperparams
def save_config_values(module, target):
    target += "_config.txt"
    if not tf.gfile.Exists(target):
        vars = {}
        if module:
            vars = {key: value for key, value in module.__dict__.items() if
                    not (key.startswith('__') or key.startswith('_'))}
            file = open(target, "a")
            for key, value in vars.items():
                file.write(str(key) + " : " + str(value) + "\n")


# download data set from url and save to filesystem if not present yet
def download_data(url, target):
    urlretrieve(url, target + "/data.zip")
    with zipfile.ZipFile(target + "/data.zip", "r") as zip_ref:
        zip_ref.extractall(target)


def _to_valid_filename(str_):
    str_ = re.sub('[^\w\s-]', '', str_).strip().lower()
    str_ = re.sub('[-\s]+', '-', str_)

    if len(str_) > 200:
        str_ = str_[:100] + str_[len(str_) - 100:]

    return str_


# create attention visualization and save as plot image
def plot_attention(value, a_words, title_text, y_lab, savepath, filename):
    value, a_words = words2chars(value, a_words)
    value = value[::-1]
    a_words = a_words[::-1]
    y_lab = y_lab[::-1]

    filename = savepath + "/" + _to_valid_filename(filename) + ".png"
    if not os.path.isfile(filename):
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        w_count = min(200, value.shape[1])
        plt.clf()
        a_words = a_words[:, :w_count]
        value = value[:, :w_count]
        x = []
        for i in range(0, w_count):
            x.append(i)

        plt.figure(figsize=(19, 8))

        heatmap = plt.pcolor(value, cmap="YlOrRd")

        for y in range(value.shape[0]):
            for z in range(value.shape[1]):
                plt.text(z + 0.5, y + 0.5, a_words[y, z],
                         horizontalalignment='center',
                         verticalalignment='center', )

        plt.colorbar(heatmap)
        plt.ylabel('sentence')
        plt.xlabel('word')
        plt.title(title_text)

        plt.autoscale(True)
        plt.xticks(x, a_words[0], rotation='vertical')
        plt.yticks(range(0, value.shape[0]), y_lab)
        plt.axes().get_xaxis().set_visible(False)
        # plt.show()
        plt.savefig(filename)
        plt.close()


# helper function for visualization, convert words to chars with values for more fine-grained grid in heatmap
def words2chars(values, words):
    sents = []
    char_vals = []
    for i, sent in enumerate(words):
        chars = []
        char_val = []
        val = values[i]
        for j, word in enumerate(sent):
            value = val[j]
            for k, ch in enumerate(word):
                chars.append(ch)
                char_val.append(value)
            chars.append(" ")
            char_val.append(value)

        char_vals.append(char_val)
        sents.append(chars)
    chars_pad = np.empty([len(sents), len(max(sents, key=lambda x: len(x)))], dtype=np.chararray)
    chars_pad[:] = ''
    vals_pad = np.zeros_like(chars_pad, dtype=np.float32)
    for i, j in enumerate(sents):
        chars_pad[i][0:len(j)] = j
    for i, j in enumerate(char_vals):
        vals_pad[i][0:len(j)] = j
    return vals_pad, chars_pad


# mean average precision for batch
def average_precision(probabs, labels, a_counts):
    m_ap = 0.0
    for i, lab in enumerate(labels):
        ap = example_precision(logits=probabs[i], labels=labels[i], a_count=a_counts[i])
        m_ap += ap
    m_ap = m_ap / len(labels)
    return np.float32(m_ap)


# mean reciprocal rank for batch
def average_rank(probabs, labels, a_counts):
    m_ar = 0.0
    for i, lab in enumerate(labels):
        ar = example_rank(logits=probabs[i], labels=labels[i], a_count=a_counts[i])
        # print(ap)
        m_ar += ar
    m_ar = m_ar / len(labels)
    return np.float32(m_ar)


# mean reciprocal rank for single sample
def example_rank(logits, labels, a_count):
    labels = labels[:a_count]
    logits = logits[:a_count]
    mrr = 0
    extracted = {}
    for i, label in enumerate(labels):
        if label > 0.0:
            extracted[i] = 1
    indices = np.argsort(logits)[::-1]
    for j, index in enumerate(indices):
        if index in extracted:
            mrr = 1 / (j + 1)
            break
    if (mrr > 0):
        return mrr
    else:
        return 0.0


# mean average precision for single sample
def example_precision(logits, labels, a_count):
    labels = labels[:a_count]
    logits = logits[:a_count]
    map_idx = 0
    map_val = 0
    extracted = {}
    for i, label in enumerate(labels):
        if label > 0.0:
            extracted[i] = 1
    indices = np.argsort(logits)[::-1]
    for j, index in enumerate(indices):
        if index in extracted:
            map_idx = map_idx + 1
            map_val = map_val + (map_idx / (j + 1))
    if (map_idx > 0):
        map_val = map_val / map_idx
        return map_val
    else:
        return 0.0


def print_predictions(outfolder, step, gold, predicted_probabilities, split):
    prediction = np.argmax(predicted_probabilities)
    correct = int(prediction == gold)

    if split == "val":
        line = "question %d\t%d\t%d\t%d\t%s" % (step, gold, prediction, correct, str(predicted_probabilities))
    else:
        line = "question %d\t%d\t%s" % (step, prediction, str(predicted_probabilities))

    # header, overwrite old file
    if step == 0:
        mode = "w"
        if split == "val":
            header = "question\tgold\tpredicted\tcorrect\tpredicted probabilities"
        else:
            header = "question\tpredicted\tpredicted probabilities"

        line = "%s\n%s" % (header, line)
    # append info to file
    else:
        mode = "a"

    with open(outfolder + "/probabilities.txt", mode) as file:
        file.write(line + "\n")


def print_sentence_attentions(outfolder, step, attention):
    # overwrite old file
    if step == 0:
        mode = "w"
    # append to file
    else:
        mode = "a"

    with open(outfolder + "/attentions.txt", mode) as file:
        # write header
        if step == 0:
            file.write("question\tanswer\tsentence attention distribution\n")
        for i, att in enumerate(attention):
            file.write("question %d\t%d\t%s\n" % (step, i, str(attention[i])))
