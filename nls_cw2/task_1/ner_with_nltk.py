"""
process corpus_1 (inaugural scripts) with nltk.ne_chunks()
"""

from typing import List
import nltk
from nltk import pos_tag
from nltk.corpus import inaugural
from tqdm import tqdm
from nls_cw2.paths import NER_WITH_NLTK_NDJSON
import json


def main():
    # download the corpus & pos-tagger if needed
    nltk.download('inaugural')
    nltk.download('averaged_perceptron_tagger')
    # iterate over the sentences in the Inaugural corpus
    results = list()
    for sent in tqdm(inaugural.sents()):
        sent: List[str]
        # in order for ne_chunk algorithm to work, the sentences must be pos-tagged
        sent_tagged = pos_tag(tokens=sent)
        # this returns a nltk.Tree, where the root is "S" and the ner-tagged tokens are children.
        chunks: nltk.Tree = nltk.ne_chunk(tagged_tokens=sent_tagged)
        results.append(chunks.pformat())
    # save the results
    with open(NER_WITH_NLTK_NDJSON, 'w') as fh:
        for idx, res in enumerate(results):
            if idx in (450, 4105):
                # have to hard code this due to a bug in nltk.Tree.pformat.
                to_write = json.dumps(res + ")") + "\n"
            else:
                to_write = json.dumps(res) + "\n"
            fh.write(to_write)


if __name__ == '__main__':
    main()
