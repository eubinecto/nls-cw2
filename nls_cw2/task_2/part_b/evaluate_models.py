"""
we want to compare the accuracies of:
1. baseline
2. bow classifier
3. w2v classifier
4. bow classifier with more features
5. w2v classifier with more features
"""
import argparse
from typing import Dict
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from nls_cw2.loaders import load_mpqa_lexicons, load_corpus_2
from typing import List, Tuple
import numpy as np
from gensim.models import KeyedVectors
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
    print(X.shape)
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
    print(X.shape)
    return X, labels


class BaselineClassifier:
    """
    Uses MPQA sentiment lexicon.
    """
    def __init__(self):
        self.lemma2sentiment: Dict[str, str] = load_mpqa_lexicons()

    def predict(self, sent: str) -> str:
        tokens = nltk.word_tokenize(sent)
        # look them up.
        labels = [
            self.lemma2sentiment[token]
            for token in tokens
            if token in self.lemma2sentiment.keys()
        ]
        # how many positive do we have?
        pos_cnt = len([None for label in labels if label == "positive"])
        # how many negatives do we have?
        neg_cnt = len([None for label in labels if label == "negative"])

        if pos_cnt == neg_cnt:
            return "neu"  # neutral
        elif pos_cnt > neg_cnt:
            return "pos"  # positive
        else:
            return "neg"  # negative


def evaluate_base() -> float:
    """
    evaluate the accuracy of the baseline model.
    :return:
    """
    # prepare the corpus
    corpus_2_pos = load_corpus_2(positive=True)
    corpus_2_neg = load_corpus_2(positive=False)
    # prepare the classifier
    baseline_cls = BaselineClassifier()
    # evaluate on the positive corpus
    pos_total = 0
    pos_correct = 0
    for sent in corpus_2_pos:
        pos_total += 1
        if baseline_cls.predict(sent) == "pos":
            pos_correct += 1
    # evaluate on the negative corpus
    neg_total = 0
    neg_correct = 0
    for sent in corpus_2_neg:
        neg_total += 1
        if baseline_cls.predict(sent) == "neg":
            neg_correct += 1
    # compute the acc and return
    acc = (pos_correct + neg_correct) / (pos_total + neg_total)
    return acc


HYPER_PARAMS = {
    'n_estimators': 100,
    'max_depth': 3,
    'verbose': 1,
    'n_jobs': 4,
}


def evaluate_bow(k: int, more_features: bool) -> float:
    """
    evaluate the bow model with k-fold cross validation
    :return:
    """
    global HYPER_PARAMS

    X_bow, y_bow = preproc_dataset_bow(more_features)
    clf_bow = RandomForestClassifier(**HYPER_PARAMS)
    # perform cross validation
    scores = cross_val_score(clf_bow, X_bow, y_bow, cv=k, verbose=True)
    return scores.mean()


def evaluate_w2v(k: int, more_features: bool) -> float:
    """
    evaluate the w2v model with k-fold cross validation
    :param k:
    :param more_features:
    :return:
    """
    global HYPER_PARAMS
    X_w2v, y_w2v = preproc_dataset_w2v(more_features)
    clf_w2v = RandomForestClassifier(**HYPER_PARAMS)
    # perform cross validation
    scores = cross_val_score(clf_w2v, X_w2v, y_w2v, cv=k, verbose=True)
    return scores.mean()


def main():
    # evaluate them all. be flexible here.
    parser = argparse.ArgumentParser()
    # available models: base, bow, w2v
    parser.add_argument('--model', type=str,
                        default="bow")
    # this is for bow & w2v models.
    parser.add_argument('--k', type=int,
                        default=4)  # the value of k for k-fold validation
    parser.add_argument("--more_features", dest="more_features",
                        default=False, action="store_true")   # include / not include more features.
    args = parser.parse_args()

    # --- choosing the model.
    if args.model == "base":
        acc = evaluate_base()
    elif args.model == "bow":
        acc = evaluate_bow(k=args.k, more_features=args.more_features)
    elif args.model == "w2v":
        acc = evaluate_w2v(k=args.k, more_features=args.more_features)
    else:
        raise ValueError

    print("#### EVAL RESULT ####")
    print("|".join([args.model, str(args.k), str(args.more_features)]))
    print(HYPER_PARAMS)
    print("model accuracy:", acc)


if __name__ == '__main__':
    main()
