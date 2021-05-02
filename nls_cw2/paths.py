from os import path
from os import environ
HOME_DIR = path.expanduser("~")
DATA_DIR = path.join(HOME_DIR, 'data')
PROJECT_DATA_DIR = path.join(DATA_DIR, "data_nls_cw2")
STANFORD_NER_DIR = path.join(PROJECT_DATA_DIR, "stanford-ner")  # for task 1
# the corpora
CORPUS_1_DIR = path.join(PROJECT_DATA_DIR, "corpus_1")  # for task 1
CORPUS_2_DIR = path.join(PROJECT_DATA_DIR, "corpus_2")  # for task 2


STANFORD_NER_JAR = path.join(STANFORD_NER_DIR, "stanford-ner.jar")
MODEL_7 = path.join(STANFORD_NER_DIR, "classifiers/english.muc.7class.distsim.crf.ser.gz")

# environment vars - need to set these up for things to work.
environ["CLASSPATH"] = STANFORD_NER_JAR
environ["STANFORD_MODELS"] = MODEL_7
