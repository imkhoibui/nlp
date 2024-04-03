import os
import re

from nltk.tokenize import WhitespaceTokenizer

def load_glove():
    word_2_vec = {}

    print("Loading GloVe")

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove.6B.50d-relativized.txt")) as file:
        for line in file:
            l = line.split()
            word_2_vec[l[0]] = l[1:]

    print("Finish loading GloVe")
    return word_2_vec

def read_file(data):
    data_file = []
    with open(data, "r") as file:
        for line in file:
            data_file.append(line)
    return data_file

def split_data(data):
    targets = {}
    features = {}
    with open(data, "r") as file:
        i = 0
        for lines in file:
            line = lines.split("\t")
            targets[i] = line[0]
            features[i] = tokenizer(line[1])
    return features, targets

def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w]', ' ', sentence)
    return sentence.replace("  ", " ")

def tokenizer(sentence):
    sentence = clean_sentence(sentence)
    return WhitespaceTokenizer().tokenize(sentence)