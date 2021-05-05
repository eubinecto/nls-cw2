"""
collect sentiment lexicons with the basic patterns (i.e. conjoined with and, conjoined with but).
"""
import csv
import string

from nltk import word_tokenize, pos_tag, RegexpParser, Tree
from nltk.corpus import stopwords
from tqdm import tqdm
from nls_cw2.loaders import load_adjs_init, load_corpus_2
from nls_cw2.paths import *
from itertools import chain


# adj-conjoined-adj pattern.
CHUNK_PATTERN = """
CONJOIN: {<JJ><CC><JJ>}
"""
EXC_LIST = string.punctuation + string.digits


def main():
    # use REGEXP parser to correctly collect only the adjectives.
    # the pattern should be - <adj><con><adj>
    # first, load initial lexicons.
    adjs_init = load_adjs_init()
    positives_init = [adj for adj, senti in adjs_init if senti == "pos"]
    negatives_init = [adj for adj, senti in adjs_init if senti == "neg"]
    print(positives_init)
    print(negatives_init)
    # load corpus 2
    all_sents = chain(load_corpus_2(positive=True), load_corpus_2(positive=False))
    # the targets. positives_found, and negatives.
    positives_found = list()
    negatives_found = list()
    # init a parser with the pattern
    parser = RegexpParser(CHUNK_PATTERN)
    for sent in tqdm(all_sents):
        tokens = word_tokenize(sent)
        tree = pos_tag(tokens)
        for child in parser.parse(tree):
            if isinstance(child, Tree):
                if child.label() == "CONJOIN":
                    # get only the adjectives that were found.
                    founds = [
                        token
                        for token, pos in child
                        if token not in EXC_LIST
                        if pos != "CC"
                    ]
                    # if any of the positive lexicons is here, add found to positive founds.
                    for pos_adj in positives_init:
                        if pos_adj in founds:
                            for found_token in founds:
                                positives_found.append(found_token)
                            break
                    else:
                        # else, if any of the negative lexicons are here, add found to negative founds
                        for neg_adj in negatives_init:
                            if neg_adj in founds:
                                for found_token in founds:
                                    negatives_found.append(found_token)
    # compute the frequencies - used later when assigning polarities.
    pos_token2cnt = dict()
    neg_token2cnt = dict()
    for pos_token, neg_token in zip(positives_found, negatives_found):
        pos_token2cnt[pos_token] = pos_token2cnt.get(pos_token, 0) + 1
        neg_token2cnt[neg_token] = neg_token2cnt.get(neg_token, 0) + 1

    # save them
    with open(ADJS_BASIC_TSV, 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter="\t")
        for pos_adj, freq in pos_token2cnt.items():
            tsv_writer.writerow([pos_adj, 'pos', freq])
        for neg_adj, freq in neg_token2cnt.items():
            tsv_writer.writerow([neg_adj, 'neg', freq])


if __name__ == '__main__':
    main()
