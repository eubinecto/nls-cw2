"""
evaluate the accuracy of the baseline model.
"""
from nls_cw2.classifiers import BaselineClassifier, Category
from nls_cw2.loaders import load_corpus_2


def main():
    # prepare the corpus
    corpus_2_pos = load_corpus_2(positive=True)
    corpus_2_neg = load_corpus_2(positive=False)

    # prepare the classifier
    baseline_cls = BaselineClassifier()

    # evaluate on the positive corpus
    pos_total = 0
    pos_correct = 0
    for sent in corpus_2_pos:
        pos_total += 1
        if baseline_cls(sent) == Category.POS:
            pos_correct += 1

    # evaluate on the negative corpus
    neg_total = 0
    neg_correct = 0
    for sent in corpus_2_pos:
        neg_total += 1
        if baseline_cls(sent) == Category.NEG:
            neg_correct += 1

    acc = (pos_correct + neg_correct) / (pos_total + neg_total)
    print("baseline accuracy:", acc)


if __name__ == '__main__':
    main()
