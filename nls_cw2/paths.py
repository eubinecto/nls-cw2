from pathlib import Path
from os import path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = path.join(ROOT_DIR, "data")
# the corpora
CORPUS_1_DIR = path.join(DATA_DIR, "corpus_1")  # for task 1
CORPUS_2_DIR = path.join(DATA_DIR, "corpus_2")  # for task 2

