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


def load_adjs_init() -> List[Tuple[str, str]]:
    with open(ADJS_INIT_TSV, 'r') as fh:
        tsv_reader = csv.reader(fh, delimiter="\t")
        return [
            (row[0].strip().replace("\ufeff", ""), row[1].strip())
            for row in tsv_reader
        ]


def load_adjs_basic() -> List[Tuple[str, str, int]]:
    with open(ADJS_BASIC_TSV, 'r') as fh:
        tsv_reader = csv.reader(fh, delimiter="\t")
        return [
            (row[0], row[1], int(row[2]))
            for row in tsv_reader
        ]


def load_adjs_more() -> List[Tuple[str, str, int]]:
    with open(ADJS_MORE_TSV, 'r') as fh:
        tsv_reader = csv.reader(fh, delimiter="\t")
        return [
            (row[0], row[1], int(row[2]))
            for row in tsv_reader
        ]


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


def load_dataset() -> Generator[Tuple[str, int, str], None, None]:
    with open(DATASET_TSV, 'r') as fh:
        tsv_reader = csv.reader(fh, delimiter="\t")
        for row in tsv_reader:
            yield row[0], int(row[1]), row[2]
