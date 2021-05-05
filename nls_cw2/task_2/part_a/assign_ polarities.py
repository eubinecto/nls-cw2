"""
wait, so we already have positives and negatives. now what do we want to do?
we want to know how positive and negative they are.
"""
import csv

from nltk import word_tokenize
from nls_cw2.loaders import load_adjs, load_corpus_2
from nls_cw2.paths import *


def scaled_likelihood(word: str, sentiment: str) -> float:
    """
    p(w|c) = f(w, c) / f(w, c) for all w in c.
    p(w|c) / p(w)
    :param word:
    :param sentiment:
    :return:
    """
    if sentiment == "pos":
        corpus = load_corpus_2(positive=True)
    elif sentiment == "neg":
        corpus = load_corpus_2(positive=False)
    else:
        raise ValueError
    c_tokens = [
        token
        for sent in corpus
        for token in word_tokenize(sent)
    ]
    c_token2cnt = dict()
    for token in c_tokens:
        c_token2cnt[token] = c_token2cnt.get(token, 0) + 1
    f_w_c = c_token2cnt[word]  # must exist
    all_f_w_c = ...
    p_w_bar_c = f_w_c / all_f_w_c
    p_w = ...
    scaled = p_w_bar_c / p_w
    return scaled


def main():
    # first, load all positive & negative adjectives.
    pos_init, neg_init = load_adjs("init")
    pos_basic, neg_basic = load_adjs("basic")
    pos_more, neg_more = load_adjs("more")
    pos_all = pos_init + pos_basic + pos_more
    neg_all = neg_init + neg_basic + neg_more
    # load positive and negative sentences, and build the vocabularies
    pos_sents = load_corpus_2(positive=True)
    neg_sents = load_corpus_2(positive=False)
    pos_words = [
        token
        for sent in pos_sents
        for token in word_tokenize(sent)
    ]
    neg_words = [
        token
        for sent in neg_sents
        for token in word_tokenize(sent)
    ]
    #  count the tokens
    pos_token2cnt = dict()
    neg_token2cnt = dict()
    for pos_token, neg_token in zip(pos_words, neg_words):
        pos_token2cnt[pos_token] = pos_token2cnt.get(pos_token, 0) + 1
        neg_token2cnt[neg_token] = neg_token2cnt.get(neg_token, 0) + 1
    pos_vocab_size = len(pos_token2cnt.keys())
    neg_vocab_size = len(neg_token2cnt.keys())

    with open(POS_POLARS_TSV, 'w') as pos_fh, open(NEG_POLARS_TSV, 'w') as neg_fh:
        pos_tsv_writer = csv.writer(pos_fh, delimiter="\t")
        neg_tsv_writer = csv.writer(neg_fh, delimiter="\t")
        for pos_adj, neg_adj in zip(pos_all, neg_all):
            prob_pos_adj = (pos_token2cnt.get(pos_adj, 0) + neg_token2cnt.get(pos_adj, 0))\
                           / (pos_vocab_size + neg_vocab_size)
            prob_neg_adj = (pos_token2cnt.get(neg_adj, 0) + neg_token2cnt.get(neg_adj, 0)) \
                           / (pos_vocab_size + neg_vocab_size)
            pos_likelihood = pos_token2cnt.get(pos_adj, 0) / pos_vocab_size
            neg_likelihood = neg_token2cnt.get(neg_adj, 0) / neg_vocab_size
            pos_likelihood_scaled = pos_likelihood / prob_pos_adj
            neg_likelihood_scaled = neg_likelihood / prob_neg_adj
            pos_tsv_writer.writerow([pos_adj, pos_likelihood_scaled])
            neg_tsv_writer.writerow([neg_adj, neg_likelihood_scaled])


if __name__ == '__main__':
    main()
