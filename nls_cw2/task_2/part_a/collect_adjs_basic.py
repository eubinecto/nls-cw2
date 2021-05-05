"""
collect sentiment lexicons with the basic patterns (i.e. conjoined with and, conjoined with but).
"""
import string

from nltk import word_tokenize, pos_tag, RegexpParser, Tree
from nltk.corpus import stopwords
from tqdm import tqdm

from nls_cw2.loaders import load_lexicons, load_corpus_2
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
    init_lexicons = load_lexicons('init')
    positives_init = set(init_lexicons[0])
    negatives_init = set(init_lexicons[1])
    # load corpus 2
    all_sents = chain(load_corpus_2(positive=True), load_corpus_2(positive=False))
    # the targets. positives_found, and negatives.
    positives_found = set()
    negatives_found = set()
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
                                positives_found.add(found_token)
                            break
                    else:
                        # else, if any of the negative lexicons are here, add found to negative founds
                        for neg_adj in negatives_init:
                            if neg_adj in founds:
                                for found_token in founds:
                                    negatives_found.add(found_token)

    # just get the new ones.
    positives_new = positives_found - positives_init
    negatives_new = negatives_found - negatives_init
    # filter stopwords
    stop_words = set(stopwords.words('english'))
    positives_new = [token for token in positives_new if token not in stop_words]
    negatives_new = [token for token in negatives_new if token not in stop_words]

    # save them
    with open(BASIC_POS_TXT, 'w') as fh_pos, open(BASIC_NEG_TXT, 'w') as fh_neg:
        for pos_lex, neg_lex in zip(positives_new, negatives_new):
            fh_pos.write(pos_lex + "\n")
            fh_neg.write(neg_lex + "\n")


if __name__ == '__main__':
    main()
