from os import path
from os import environ
HOME_DIR = path.expanduser("~")
DATA_DIR = path.join(HOME_DIR, 'data')
PROJECT_DATA_DIR = path.join(DATA_DIR, "data_nls_cw2")
STANFORD_NER_DIR = path.join(PROJECT_DATA_DIR, "stanford-ner")  # for task 1
CORPUS_1_DIR = path.join(PROJECT_DATA_DIR, "corpus_1")
CORPUS_2_DIR = path.join(PROJECT_DATA_DIR, "corpus_2")

# task 1
STANFORD_NER_JAR = path.join(STANFORD_NER_DIR, "stanford-ner.jar")
STAN_MODEL_7_GZ = path.join(STANFORD_NER_DIR, "classifiers/english.muc.7class.distsim.crf.ser.gz")
NER_WITH_NLTK_NDJSON = path.join(CORPUS_1_DIR, 'ner_with_nltk.ndjson')  # ner-processed corpus 1
NER_WITH_STAN_NDJSON = path.join(CORPUS_1_DIR, 'ner_with_stan.ndjson')  # ner-processed corpus 1


# task 2


# environment vars - need to set these up for things to work.
environ["CLASSPATH"] = STANFORD_NER_JAR
environ["STANFORD_MODELS"] = STAN_MODEL_7_GZ
