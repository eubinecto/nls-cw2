import json
import nltk
from nltk.corpus import inaugural
from nls_cw2.paths import STAN_MODEL_7_GZ, NER_WITH_STAN_NDJSON


def main():
    # download the corpus if not downloaded yet
    nltk.download('inaugural')
    # instantiate the Stanford NER tagger. This is a python handler provided by nltk.
    st = nltk.StanfordNERTagger(STAN_MODEL_7_GZ)
    # ner-process all the sentences
    results = st.tag_sents(list(inaugural.sents()))
    # save the results
    with open(NER_WITH_STAN_NDJSON, 'w') as fh:
        for res in results:
            fh.write(json.dumps(res) + "\n")


if __name__ == '__main__':
    main()
