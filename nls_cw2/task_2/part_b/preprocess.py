from typing import List, Tuple
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nls_cw2.loaders import load_dataset
from nls_cw2.paths import GLOVE_BIN
from nltk import word_tokenize, WordNetLemmatizer

STOP_WORDS = set(stopwords.words('english'))


def preproc_sents() -> List[List[str]]:
    global STOP_WORDS
    lemmatizer = WordNetLemmatizer()
    sents: List[str] = [sent for sent, _, _ in load_dataset()]
    docs: List[List[str]] = [word_tokenize(sent) for sent in sents]
    docs_cleansed = [
        [
            lemmatizer.lemmatize(token)
            for token in doc
            if token not in STOP_WORDS
        ]
        for doc in docs
    ]
    return docs_cleansed


def preproc_dataset_bow(more_features: bool) -> Tuple[np.array, List[str]]:
    """
    :return: X(bow;contains_negation), y(a vector of 1's (positive) and 0's (negative))
    """
    # load all the data
    docs = preproc_sents()
    negations: List[int] = [negation for _, negation, _ in load_dataset()]
    labels: List[str] = [label for _, _, label in load_dataset()]
    # tokenise the sents
    raw_docs = [" ".join(doc) for doc in docs]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(raw_docs).toarray()
    if more_features:
        # add the contains_negation column to the features.
        X_negation = np.reshape(np.array(negations), newshape=(len(negations), 1))
        X = np.concatenate((X, X_negation), axis=1)
    print(X)
    return X, labels


def preproc_dataset_w2v(more_features: bool) -> Tuple[np.array, List[str]]:
    # load all the data
    docs = preproc_sents()
    negations: List[int] = [negation for _, negation, _ in load_dataset()]
    labels: List[str] = [label for _, _, label in load_dataset()]
    # tokenise and get the word vectors
    glove_kv = KeyedVectors.load_word2vec_format(GLOVE_BIN, binary=True)
    X = np.array([
        # get the word vectors for each token,
        np.array([
            # get the word vectors for each token
            glove_kv.get_vector(token)
            for token in doc
            # if word2vec has not seen the key, ignore it. (one of the limits of word2vec approach )
            if glove_kv.key_to_index.get(token, None)
        ]).mean(axis=0)  # average them out.
        for doc in docs
    ])
    # load pre-trained word vectors
    if more_features:
        # add the contains_negation column to the features.
        X_negation = np.reshape(np.array(negations), newshape=(len(negations), 1))
        X = np.concatenate((X, X_negation), axis=1)
    print(X)
    return X, labels
