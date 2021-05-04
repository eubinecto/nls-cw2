"""
Measure complete and partial overlap.
"""
from typing import List, Tuple

import nltk
from nls_cw2.loaders import load_ner_with_nltk, load_ner_with_stan


def reduce_to_leaves(tree: nltk.Tree) -> List[str]:
    """
    traverse the tree recursively to get the leaves,
    while labeling the leaves with all of its parents' labels.
    :param tree:
    :return:
    """
    leaves = list()
    for child in tree:
        # recursive call
        if isinstance(child, nltk.Tree):
            leaves += reduce_to_leaves(child)
        else:
            # label the leaf with the parent's label
            leaves.append(child + "+" + tree.label())
    return leaves


def main():
    sents_tagged_nltk = load_ner_with_nltk()
    sents_tagged_stan = load_ner_with_stan()
    sent2orgidxs_nltk: List[List[int]] = list()
    for s_idx, sent in enumerate(sents_tagged_nltk):
        sent: nltk.Tree
        leaves = reduce_to_leaves(sent)
        orgidxs = [
            l_idx
            for l_idx, leave in enumerate(leaves)
            if leave.split("+")[-1] == "ORGANIZATION"
        ]
        sent2orgidxs_nltk.append(orgidxs)
    sent2orgidxs_stan: List[List[int]] = list()
    for s_idx, sent in enumerate(sents_tagged_stan):
        sent: List[Tuple[str, str]]
        # just get the indices of organization-tagged tokens.
        orgidxs = [idx for idx, (_, tag) in enumerate(sent) if tag == "ORGANIZATION"]
        sent2orgidxs_stan.append(orgidxs)

    print(sent2orgidxs_nltk)
    print(sent2orgidxs_stan)
    # now find the exact matches & partial matches
    exact_match = 0
    partial_match = 0
    for orgidxs_nltk, orgidxs_stan in zip(sent2orgidxs_nltk, sent2orgidxs_stan):
        if not orgidxs_nltk or not orgidxs_stan:
            continue
        if orgidxs_nltk == orgidxs_stan:
            exact_match += 1  # complete agreement
        else:
            min_nltk = orgidxs_nltk[0]
            min_stan = orgidxs_stan[0]
            max_nltk = orgidxs_nltk[-1]
            max_stan = orgidxs_stan[-1]
            if (min_stan <= min_nltk) and (max_nltk <= max_stan):
                partial_match += 1
            elif (min_nltk <= min_stan) and (max_nltk <= max_stan):
                partial_match += 1

    print("exact match:", exact_match)
    print("partial match:", partial_match)


if __name__ == '__main__':
    main()
