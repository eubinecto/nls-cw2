import csv
import json
from typing import List, Tuple, Generator, Dict
import nltk
# just import all the paths
from nls_cw2.paths import *


# --- task 1 --- #
def load_ner_with_nltk() -> Generator[List[nltk.Tree], None, None]:
    with open(NER_WITH_NLTK_NDJSON, 'r') as fh:
        for line in fh:
            # load as nltk Tree's
            yield nltk.Tree.fromstring(json.loads(line))


def load_ner_with_stan() -> Generator[List[Tuple[str, str]], None, None]:
    with open(NER_WITH_STAN_NDJSON, 'r') as fh:
        for line in fh:
            yield json.loads(line)


# --- task 2 --- #
def load_corpus_2(positive: bool) -> Generator[str, None, None]:
    txt_path = RT_POS_TXT if positive else RT_NEG_TXT
    with open(txt_path, 'r', encoding="ISO-8859-1") as fh:
        for line in fh:
            yield line.strip()


def load_lexicons(kind: str) -> Tuple[List[str], List[str]]:
    """
    :param kind:
    :return: Tuples. lexicons at index 0 are positive, those at index 1 are negative.
    """
    if kind == "init":
        pos_txt = INIT_POS_TXT
        neg_txt = INIT_NEG_TXT
    elif kind == "basic":
        pos_txt = BASIC_POS_TXT
        neg_txt = BASIC_NEG_TXT
    elif kind == "more":
        pos_txt = MORE_POS_TXT
        neg_txt = MORE_NEG_TXT
    else:
        raise ValueError
    with open(pos_txt, 'r') as fh_pos, open(neg_txt, 'r') as fh_neg:
        positives = [line.strip() for line in fh_pos]
        negatives = [line.strip() for line in fh_neg]
    return positives, negatives


def load_mpqa_lexicons() -> Dict[str, str]:
    with open(SUBJ_CLUES_TFF, 'r') as fh:
        # the file is space-separated
        lemma2sentiment = dict()
        ssv_reader = csv.reader(fh, delimiter=" ")
        for row in ssv_reader:
            lemma = row[2].split("=")[-1]
            sentiment = row[-1].split("=")[-1]
            lemma2sentiment[lemma] = sentiment
        return lemma2sentiment
