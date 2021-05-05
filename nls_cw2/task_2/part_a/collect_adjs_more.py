"""
when adjectives are enumerated with commas, they are likely to share similar sentiments.
e.g.1.  one of the greatest family-oriented , fantasy-adventure movies ever .
e.g.2.  a thoughtful , provocative , insistently humanizing film .
e.g.3.  guaranteed to move anyone who ever shook , rattled , or rolled .
"""
from itertools import chain
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

from nls_cw2.paths import *
from nls_cw2.loaders import load_lexicons, load_corpus_2
import re
import string
# comma-enumeration pattern.
CHUNK_PATTERN = """
ENUM: {<JJ><,><JJ><,><JJ><,><JJ>}
      {<JJ><,><JJ><,><JJ>}
      {<JJ><,><JJ>}  
"""
EXC_LIST = string.punctuation + string.digits


def main():
    global CHUNK_PATTERN, EXC_LIST
    # first, load initial lexicons.
    init_lexicons = load_lexicons('init')
    positives_init = set(init_lexicons[0])
    negatives_init = set(init_lexicons[1])
    # load corpus 2
    all_sents = chain(load_corpus_2(positive=True), load_corpus_2(positive=False))
    # init parser with the pattern
    parser = nltk.RegexpParser(CHUNK_PATTERN)
    positives_found = set()
    negatives_found = set()
    for sent in tqdm(all_sents):
        tokens = nltk.word_tokenize(sent)
        tree = nltk.pos_tag(tokens)
        for child in parser.parse(tree):
            if isinstance(child, nltk.Tree):
                if child.label() == "ENUM":
                    founds = [token for token, pos in child if token not in EXC_LIST]
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
    # get the new one's and save.
    positives_new = positives_found - positives_init
    negatives_new = negatives_found - negatives_init
    # filter stopwords
    stop_words = set(stopwords.words('english'))
    positives_new = [token for token in positives_new if token not in stop_words]
    negatives_new = [token for token in negatives_new if token not in stop_words]
    # save them
    with open(MORE_POS_TXT, 'w') as fh_pos, open(MORE_NEG_TXT, 'w') as fh_neg:
        for pos_lex, neg_lex in zip(positives_new, negatives_new):
            fh_pos.write(pos_lex + "\n")
            fh_neg.write(neg_lex + "\n")


if __name__ == '__main__':
    main()
