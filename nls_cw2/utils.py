from typing import List
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


def cleanse(tokens: List[str]) -> List[str]:
    global stop_words
    # TODO: remove stopwords. remove punctuations.
    return [
        token
        for token in tokens
        if token not in stop_words
    ]
