"""
Need to implement three classifiers.
"""
from abc import ABC
from typing import List, Dict
from nls_cw2.loaders import load_mpqa_lexicons
from enum import Enum, auto


class Category(Enum):
    NEU = auto()  # neutral
    POS = auto()  # positive
    NEG = auto()  # negative


class Classifier:

    def __call__(self, *args, **kwargs) -> Category:
        """
        given an input, classifies it into a category.
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def tokenise(sent: str) -> List[str]:
        return sent.split(" ")


class BaselineClassifier(Classifier):
    """
    Uses MPQA sentiment lexicon.
    """
    def __init__(self):
        self.lemma2sentiment: Dict[str, str] = load_mpqa_lexicons()

    def __call__(self, sent: str) -> Category:
        tokens = self.tokenise(sent)
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
            return Category.NEU  # neutral
        elif pos_cnt > neg_cnt:
            return Category.POS  # positive
        else:
            return Category.NEG  # negative


class MLClassifier(Classifier, ABC):

    def __init__(self):
        # TODO: use a RandomForestClassifier. No need to use a complicated model with this.
        pass

    def fit(self, sents: List[str]):
        raise NotImplementedError


class BOWClassifier(MLClassifier):

    def __call__(self, *args, **kwargs) -> Category:
        pass

    # TODO: use a one-hot vectorizer maybe?
    def fit(self, sents: List[str]):
        pass


class W2VClassifier(MLClassifier):
    # TODO: As for this, use pre-trained weights.
    def fit(self):
        pass
