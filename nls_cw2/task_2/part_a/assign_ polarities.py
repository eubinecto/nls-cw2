"""
wait, so we already have positives and negatives. now what do we want to do?
we want to know how positive and negative they are.
"""
import csv
from nls_cw2.loaders import load_adjs_basic, load_adjs_more
from nls_cw2.paths import *


def main():
    pos_adj2cnt_basic = [(adj, cnt) for adj, senti, cnt in load_adjs_basic() if senti == "pos"]
    pos_adjs_more = [(adj, cnt) for adj, senti, cnt in load_adjs_more() if senti == "pos"]
    pos_adj2freq = dict(pos_adj2cnt_basic)  # need this. f(w, pos)
    for adj, cnt in pos_adjs_more:
        pos_adj2freq[adj] = pos_adj2freq.get(adj, 0) + cnt

    neg_adj2cnt_basic = [(adj, cnt) for adj, senti, cnt in load_adjs_basic() if senti == "neg"]
    neg_adjs_more = [(adj, cnt) for adj, senti, cnt in load_adjs_more() if senti == "neg"]
    neg_adj2freq = dict(neg_adj2cnt_basic)  # need this. f(w, neg)
    for adj, cnt in neg_adjs_more:
        neg_adj2freq[adj] = neg_adj2freq.get(adj, 0) + cnt

    # the likelihoods of the positive adjectives
    pos_all_cnt = sum(pos_adj2freq.values())
    neg_all_cnt = sum(neg_adj2freq.values())

    with open(POLARITIES_TSV, 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter="\t")
        # write the polarities of the positive adjectives
        rows = sorted([
            [adj, "pos", cnt / pos_all_cnt]  # adj, c, p(adj|c)
            for adj, cnt in pos_adj2freq.items()],
            key=lambda x: x[2],
            reverse=True)
        tsv_writer.writerows(rows)

        rows = sorted([
            [adj, "neg", cnt / neg_all_cnt]  # adj, c, p(adj|c)
            for adj, cnt in neg_adj2freq.items()],
            key=lambda x: x[2],
            reverse=True)
        tsv_writer.writerows(rows)


if __name__ == '__main__':
    main()
