"""
schema: sent, sentiment (either neg, pos)
"""
import csv
from nls_cw2.loaders import load_corpus_2
from nls_cw2.paths import *
import nltk


def contains_negation(sent: str) -> int:
    tokens = nltk.word_tokenize(sent)
    for token in tokens:
        if token in ("n't", "not", "no"):
            return 1
    return 0


def main():
    corpus_2_pos = load_corpus_2(positive=True)
    corpus_2_neg = load_corpus_2(positive=False)

    # save the  data as tsv.
    with open(DATASET_TSV, 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter="\t")
        for pos_sent in corpus_2_pos:
            to_write = [pos_sent, contains_negation(pos_sent), "pos"]
            tsv_writer.writerow(to_write)
        for neg_sent in corpus_2_neg:
            to_write = [neg_sent, contains_negation(neg_sent), "neg"]
            tsv_writer.writerow(to_write)


if __name__ == '__main__':
    main()

