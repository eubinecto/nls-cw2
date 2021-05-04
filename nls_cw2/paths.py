from os import path
from os import environ
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = path.join(PROJECT_DIR, 'data')
STANFORD_NER_DIR = path.join(DATA_DIR, "stanford-ner")
CORPUS_1_DIR = path.join(DATA_DIR, "corpus_1")
CORPUS_2_DIR = path.join(DATA_DIR, "corpus_2")
LEXICONS_DIR = path.join(DATA_DIR, "lexicons")
MPQA_DIR = path.join(LEXICONS_DIR, 'mpqa')

# task 1
STANFORD_NER_JAR = path.join(STANFORD_NER_DIR, "stanford-ner.jar")
STAN_MODEL_7_GZ = path.join(STANFORD_NER_DIR, "classifiers/english.muc.7class.distsim.crf.ser.gz")
NER_WITH_NLTK_NDJSON = path.join(CORPUS_1_DIR, 'ner_with_nltk.ndjson')  # ner-processed corpus 1
NER_WITH_STAN_NDJSON = path.join(CORPUS_1_DIR, 'ner_with_stan.ndjson')  # ner-processed corpus 1


# task 2
INIT_POS_TXT = path.join(LEXICONS_DIR, "init_pos.txt")
INIT_NEG_TXT = path.join(LEXICONS_DIR, "init_neg.txt")
BASIC_POS_TXT = path.join(LEXICONS_DIR, "basic_pos.txt")
BASIC_NEG_TXT = path.join(LEXICONS_DIR, "basic_neg.txt")
MORE_POS_TXT = path.join(LEXICONS_DIR, "more_pos.txt")
MORE_NEG_TXT = path.join(LEXICONS_DIR, "more_neg.txt")
RT_POS_TXT = path.join(CORPUS_2_DIR, 'rt-polarity.pos')
RT_NEG_TXT = path.join(CORPUS_2_DIR, 'rt-polarity.neg')
SUBJ_CLUES_TFF = path.join(MPQA_DIR, 'subjclueslen1-HLTEMNLP05.tff')


# environment vars - need to set these up for things to work.
environ["CLASSPATH"] = STANFORD_NER_JAR
environ["STANFORD_MODELS"] = STAN_MODEL_7_GZ
