import os
import re

import numpy as np
from nltk.tokenize import WhitespaceTokenizer

def load_glove():
    """
        This function loads the GloVe file.
        Returns:
        - word_2_vec (dict): The GloVe file.
    """
    word_2_vec = {}

    print("Loading GloVe")

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove.6B.50d-relativized.txt")) as file:
        for line in file:
            l = line.split()
            word_2_vec[l[0]] = l[1:]

    print("Finish loading GloVe")
    return word_2_vec

def read_file(data):
    """
        This function reads the data file.
        ----------
        Parameters:
        - data (str): The path to the data file (train.txt or dev.txt).
        Returns:
        - data_file (list): The data file.
    """
    data_file = []
    with open(data, "r") as file:
        for line in file:
            data_file.append(line)
    return data_file

def split_line(sentence):
    """
        This function splits the data into features and targets.
        ----------
        Parameters:
        - sentence (str): The sentence of the data list to be split
        
        Returns:
        - features (dict): The features.
        - targets (dict): The targets.
    """
    line = sentence.split("\t")
    targets = line[0]
    features = tokenizer(line[1])
    return features, targets

def clean_sentence(sentence):
    """
        This function cleans the sentence by lowering all the capitals and remove
        all symbols.
        ----------
        Parameters:
        - sentence (str): The sentence to be cleaned.
        
        Returns:
        - sentence (str): The cleaned sentence.
    """
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w]', ' ', sentence)
    return sentence.replace("  ", " ")

def tokenizer(sentence):
    """
        This function tokenizes the sentence.
        ----------
        Parameters:
        - sentence (str): The sentence to be tokenized.
        Returns:
        - WhitespaceTokenizer().tokenize(sentence): The tokenized sentence.
    """
    sentence = clean_sentence(sentence)
    return WhitespaceTokenizer().tokenize(sentence)

def word_to_vec(word, word_2_vec):
    """
        This function vectorizes the word using word_2_vec embeddings.
        ----------
        Parameters:
        - word (str): The word to be vectorized.
        - word_2_vec (dict): The GloVe file to get word embeddings.
        Returns:
        - embed: The vectorized word.
    """
    if word in word_2_vec:
        embed = word_2_vec[word]
    else:
        embed = random_vectorize(word, word_2_vec)
    return embed

def random_vectorize(word, word_2_vec, word_vector_size=50):
    """
        This function handles the case where the word is not in the GloVe file.
        ----------
        Parameters:
        - word (str): The word to be vectorized.
        - word_2_vec (dict): The GloVe file to get word embeddings
        - word_vector_size (int): The size of the word vector (default 50)

        Returns:
        - The vectorized word.
    """
    word_2_vec[word] = np.random.uniform(0.0, 1.0, word_vector_size)
    return word_2_vec[word]
