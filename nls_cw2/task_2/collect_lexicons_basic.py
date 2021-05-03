"""
collect sentiment lexicons with the basic patterns (i.e. conjoined with and, conjoined with but).
"""
from nls_cw2.loaders import load_lexicons, load_corpus_2
from nls_cw2.paths import *
from itertools import chain


def main():
    # first, load initial lexicons.
    init_lexicons = load_lexicons('init')
    positives_init = set(init_lexicons[0])
    negatives_init = set(init_lexicons[1])
    # load corpus 2
    all_sents = chain(load_corpus_2(pos=True), load_corpus_2(pos=False))
    # the targets. positives_found, and negatives.
    positives_found = set()
    negatives_found = set()
    for sent in all_sents:
        for idx, token in enumerate(sent):
            # we skip the edge cases. There would be nothing to add.
            if idx in (0, len(sent) - 1):
                continue
            prev_token = sent[idx - 1]
            next_token = sent[idx + 1]
            # we skip if any of them are non-characters
            if token == "and":
                # those that are conjoined by and are likely to
                # express a similar sentiment to each other
                if prev_token in positives_init:
                    positives_found.add(next_token)
                elif prev_token in negatives_init:
                    negatives_found.add(next_token)
                if next_token in positives_init:
                    positives_found.add(prev_token)
                elif next_token in negatives_init:
                    negatives_found.add(prev_token)
            elif token == "but":
                # those that are conjoined by and are likely to
                # express an opposite sentiment to each other
                if prev_token in positives_init:
                    negatives_found.add(next_token)
                elif prev_token in negatives_init:
                    positives_found.add(next_token)
                if next_token in positives_init:
                    negatives_found.add(prev_token)
                elif next_token in negatives_init:
                    positives_found.add(prev_token)
    # just get the new ones.
    positives_new = positives_found - positives_init
    negatives_new = negatives_found - negatives_init

    # save them
    with open(BASIC_POS_TXT, 'w') as fh_pos, open(BASIC_NEG_TXT, 'w') as fh_neg:
        for pos_lex, neg_lex in zip(positives_new, negatives_new):
            fh_pos.write(pos_lex + "\n")
            fh_neg.write(neg_lex + "\n")


if __name__ == '__main__':
    main()
