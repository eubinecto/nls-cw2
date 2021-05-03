from typing import List, Tuple
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
    num_sent = len([None for _ in inaugural.sents()])
    for sent in tqdm(inaugural.sents(), total=num_sent):
        sent: List[str]
        # in order for ne_chunk algorithm to work, the sentences must be pos-tagged
        sent_tagged = pos_tag(tokens=sent)
        # this returns a nltk.Tree, where the root is "S" and the ner-tagged tokens are children.
        chunks: nltk.Tree = nltk.ne_chunk(tagged_tokens=sent_tagged)
        # convert the tree to a JSON-serializable format
        processed: List[Tuple] = [tuple(chunk) for chunk in chunks]
        results.append(processed)
    # save the results
    with open(NER_WITH_NLTK_NDJSON, 'w') as fh:
        for res in results:
            # as a ndjson format
            fh.write(json.dumps(res) + "\n")


if __name__ == '__main__':
    main()
