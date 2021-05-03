import json
from multiprocessing import Pool
from typing import List, Tuple
import nltk
from nltk.corpus import inaugural
from nls_cw2.paths import STAN_MODEL_7_GZ, NER_WITH_STAN_NDJSON
from tqdm import tqdm

# instantiate a stanford tagger as a global variable.
st = nltk.StanfordNERTagger(STAN_MODEL_7_GZ)


def ner_sent(sent: List[str]) -> List[Tuple[str, str]]:
    """
    use stanford ner to tag the sentence.
    :param sent:
    :return:
    """
    global st
    return st.tag(sent)


def main():
    # download the corpus if needed
    nltk.download('inaugural')
    # instantiate a stanford ner tagger. This is a python interface to the Java implementation,
    num_sents = len([None for _ in inaugural.sents()])
    # this takes quite some time, needs to be multi-processed
    with Pool(4) as p:
        # estimated take around an hour. (1.48 iterations per sec, with 4 processes.)
        results = list(tqdm(p.imap(ner_sent, inaugural.sents()), total=num_sents))
    # save the results
    with open(NER_WITH_STAN_NDJSON, 'w') as fh:
        for res in results:
            fh.write(json.dumps(res))


if __name__ == '__main__':
    main()
