import numpy as np
import numpy.typing as npt

import re


from typing import Tuple, List, Set


def bag_of_words_matrix(sentences: List[str]) -> npt.ArrayLike:
    """
    Convert the dataset into V x M matrix.
    """
    ############################# STUDENT SOLUTION ##########################
    vocab = set()
    tokenized = []
    for sentence in sentences:
        cleaned = re.sub(r"[^\w\s]", " ", sentence.lower())
        raw_tokens = cleaned.split()
        tokens = [t for t in raw_tokens if t.isalpha()]
        tokenized.append(tokens)
    

    freq = {}
    for tokens in tokenized:
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
    processed = []
    for tokens in tokenized:
        processed.append([t if freq.get(t, 0) > 2 else "<UNK>" for t in tokens])

    for tokens in processed:
        vocab.update(tokens)


    vocab = sorted(vocab)

    V = len(vocab)#number of unique tokens
    M = len(sentences)#number of sentences

    token_to_index = {}
    for i, tkn in enumerate(vocab):
        token_to_index[tkn] = i
    print(token_to_index)
    X = np.zeros((V,M), dtype=np.int32)

    for j, tokens in enumerate(tokenized):
        if not tokens:
            continue
        for tkn in tokens:
            index = token_to_index.get(tkn)
            if index is not None:
                X[index,j] += 1

    return X
    #########################################################################

def labels_matrix(data: Tuple[List[str], Set[str]]) -> npt.ArrayLike:
    """
    Convert the dataset into K x M matrix.
    """
    ############################# STUDENT SOLUTION ##########################
    sentences, labels, unique_labels = data

    sorted_labels = sorted(unique_labels)
    label_to_index = {lbl: i for i, lbl in enumerate(sorted_labels)}

    K = len(sorted_labels)  # categories
    M = len(sentences)      # number of sentences

    X = np.zeros((K, M), dtype=np.int32)

    for s, l in enumerate(labels):
        if l not in label_to_index:
            raise ValueError(f" '{l}' not found.")
        i = label_to_index[l]
        X[i, s] = 1

    return X
    #########################################################################


def softmax(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Softmax function.
    """
    ############################# STUDENT SOLUTION ##########################
    return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)
    #########################################################################


def relu(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Rectified Linear Unit function.
    """
    ############################# STUDENT SOLUTION ##########################
    return np.maximum(z,0)
    #########################################################################


def relu_prime(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    First derivative of ReLU function.
    """
    ############################# STUDENT SOLUTION ##########################
    return (z > 0).astype(z.dtype)
    #########################################################################