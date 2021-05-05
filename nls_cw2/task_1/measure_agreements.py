"""
Measure complete and partial overlap.
"""
from typing import List, Tuple

import nltk
from nls_cw2.loaders import load_ner_with_nltk, load_ner_with_stan


def main():
    sents_tagged_nltk = load_ner_with_nltk()
    sents_tagged_stan = load_ner_with_stan()

    # collect nltk-extracted organizations for each sentence.
    sent2orgs_nltk: List[List[str]] = list()
    for sent in sents_tagged_nltk:
        sent: nltk.Tree
        orgs = list()
        for child in sent:
            if isinstance(child, nltk.Tree):
                if child.label() == "ORGANIZATION":
                    org = " ".join([
                        token.split("/")[0]
                        for token in child.leaves()
                    ])
                    orgs.append(org)
        sent2orgs_nltk.append(orgs)

    # collect stanford-extracted organizations for each sentence.
    sent2orgs_stan: List[List[str]] = list()
    for sent in sents_tagged_stan:
        sent: List[Tuple[str, str]]
        # just get the indices of organization-tagged tokens.
        org_tokens = [
            token if label == "ORGANIZATION" else "*"
            for token, label in sent
        ]
        orgs = [
            token.strip()
            # space-join them, and split it by an asterisk.
            for token in " ".join(org_tokens).split("*")
            # filter out the spaces and empty strings
            if not token.isspace() and token
        ]
        sent2orgs_stan.append(orgs)

    # now find the exact matches & partial matches
    nltk_matches = sum([len(orgs) for orgs in sent2orgs_nltk])
    stan_matches = sum([len(orgs) for orgs in sent2orgs_stan])
    exact_matches = 0
    partial_matches = 0

    for orgs_nltk, orgs_stan in zip(sent2orgs_nltk, sent2orgs_stan):
        for org_n in orgs_nltk:
            for org_s in orgs_stan:
                if org_n == org_s:
                    exact_matches += 1
                    print("exact match:", org_n)
                # look for partial matches
                elif (org_n in org_s) or (org_s in org_n):
                    partial_matches += 1
                    print("partial match: {} / {}".format(org_n, org_s))

    print("nltk matches:", nltk_matches)
    print("stan matches:", stan_matches)
    print("exact matches:", exact_matches)
    print("partial matches:", partial_matches)


if __name__ == '__main__':
    main()
