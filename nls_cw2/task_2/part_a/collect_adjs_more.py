"""
when adjectives are enumerated with commas, they are likely to share similar sentiments.
e.g.1.  one of the greatest family-oriented , fantasy-adventure movies ever .
e.g.2.  a thoughtful , provocative , insistently humanizing film .
e.g.3.  guaranteed to move anyone who ever shook , rattled , or rolled .
"""
import csv
from itertools import chain
import nltk
from tqdm import tqdm
from nls_cw2.paths import *
from nls_cw2.loaders import load_corpus_2, load_adjs_init
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
    adjs_init = load_adjs_init()
    positives_init = [adj for adj, senti in adjs_init if senti == "pos"]
    negatives_init = [adj for adj, senti in adjs_init if senti == "neg"]
    # load corpus 2
    all_sents = chain(load_corpus_2(positive=True), load_corpus_2(positive=False))
    # init parser with the pattern
    parser = nltk.RegexpParser(CHUNK_PATTERN)
    positives_found = list()
    negatives_found = list()
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
    with open(ADJS_MORE_TSV, 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter="\t")
        for pos_adj, freq in pos_token2cnt.items():
            tsv_writer.writerow([pos_adj, 'pos', freq])
        for neg_adj, freq in neg_token2cnt.items():
            tsv_writer.writerow([neg_adj, 'neg', freq])


if __name__ == '__main__':
    main()
