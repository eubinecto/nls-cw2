"""
when adjectives are enumerated with commas, they are likely to share similar sentiments.
e.g.1.  one of the greatest family-oriented , fantasy-adventure movies ever .
e.g.2.  a thoughtful , provocative , insistently humanizing film .
e.g.3.  guaranteed to move anyone who ever shook , rattled , or rolled .
"""
from itertools import chain
from nls_cw2.loaders import load_lexicons, load_corpus_2
import re


def main():
    # TODO - maybe use regexp parser?
    # first, load initial lexicons.
    init_lexicons = load_lexicons('init')
    positives_init = set(init_lexicons[0])
    negatives_init = set(init_lexicons[1])
    # load corpus 2
    all_sents = chain(load_corpus_2(positive=True), load_corpus_2(positive=False))

    # find these...
    positives_found = set()
    negatives_found = set()
    for sent in all_sents:
        found = re.findall(r' [a-zA-Z\-]+? , [a-zA-Z\-]+? ', sent)
        print(sent)
        print(found)
        print("---")






if __name__ == '__main__':
    main()
