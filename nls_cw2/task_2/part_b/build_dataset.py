"""
schema: sent, sentiment (either neg =0 or pos = 1).
"""
import csv
from nls_cw2.loaders import load_corpus_2
from nls_cw2.paths import *


def main():
    corpus_2_pos = load_corpus_2(positive=True)
    corpus_2_neg = load_corpus_2(positive=False)

    # save the  data as tsv.
    with open(DATASET_TSV, 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter="\t")
        for pos_sent in corpus_2_pos:
            to_write = [pos_sent, 1]
            tsv_writer.writerow(to_write)
        for neg_sent in corpus_2_neg:
            to_write = [neg_sent, 0]
            tsv_writer.writerow(to_write)


if __name__ == '__main__':
    main()

