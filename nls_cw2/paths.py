from os import path
from os import environ
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = path.join(PROJECT_DIR, 'data')
TASK_1_DIR = path.join(DATA_DIR, "task_1")
TASK_2_DIR = path.join(DATA_DIR, "task_2")
PART_A_DIR = path.join(TASK_2_DIR, "part_a")
PART_B_DIR = path.join(TASK_2_DIR, "part_b")
STANFORD_NER_DIR = path.join(TASK_1_DIR, "stanford-ner")
CORPUS_2_DIR = path.join(TASK_2_DIR, "corpus_2")
MPQA_DIR = path.join(PART_A_DIR, 'mpqa')

# task 1 - files
STANFORD_NER_JAR = path.join(STANFORD_NER_DIR, "stanford-ner.jar")
STAN_MODEL_7_GZ = path.join(STANFORD_NER_DIR, "classifiers/english.muc.7class.distsim.crf.ser.gz")
NER_WITH_NLTK_NDJSON = path.join(TASK_1_DIR, 'ner_with_nltk.ndjson')  # ner-processed corpus 1
NER_WITH_STAN_NDJSON = path.join(TASK_1_DIR, 'ner_with_stan.ndjson')  # ner-processed corpus 1

# task 2, part a - files
RT_POS_TXT = path.join(CORPUS_2_DIR, 'rt-polarity.pos')
RT_NEG_TXT = path.join(CORPUS_2_DIR, 'rt-polarity.neg')
ADJS_BASIC_TSV = path.join(PART_A_DIR, "adjs_basic.tsv")  # w, c, f(w, c)
ADJS_MORE_TSV = path.join(PART_A_DIR, "adjs_more.tsv")  # w, c, f(w, c)
ADJS_INIT_TSV = path.join(PART_A_DIR, 'adjs_init.tsv')  # w, c
SUBJ_CLUES_TFF = path.join(MPQA_DIR, 'subjclueslen1-HLTEMNLP05.tff')
POLARS_TSV = path.join(PART_A_DIR, 'polars.tsv')  # w, c, p(w|c)

# task 2, part b - files
DATASET_TSV = path.join(PART_B_DIR, "dataset.tsv")
GLOVE_BIN = path.join(PART_B_DIR, "glove.bin")

# environment vars - need to set these up for things to work.
environ["CLASSPATH"] = STANFORD_NER_JAR
environ["STANFORD_MODELS"] = STAN_MODEL_7_GZ
