import json
from typing import List, Tuple, Generator
import nltk
from nls_cw2.paths import NER_WITH_STAN_NDJSON, NER_WITH_NLTK_NDJSON


def load_ner_with_nltk() -> Generator[List[nltk.Tree], None, None]:
    with open(NER_WITH_NLTK_NDJSON, 'r') as fh:
        for line in fh:
            # load as nltk Tree's
            yield nltk.Tree.fromstring(json.loads(line))


def load_ner_with_stan() -> Generator[List[Tuple[str, str]], None, None]:
    with open(NER_WITH_STAN_NDJSON, 'r') as fh:
        for line in fh:
            yield json.loads(line)

