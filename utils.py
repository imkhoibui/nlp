import os

def load_glove():
    word_2_vec = {}

    print("Loading GloVe")

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove.6B.50d-relativized.txt")) as file:
        for line in file:
            l = line.split()
            word_2_vec[l[0]] = l[1:]

    print("Finish loading GloVe")
    return word_2_vec