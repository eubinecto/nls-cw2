from nltk.tag import StanfordNERTagger
from nls_cw2.paths import STAN_MODEL_7_GZ
import nltk


def main():
    # stanford ner tagger.
    st = StanfordNERTagger(STAN_MODEL_7_GZ)
    sent = nltk.corpus.treebank.tagged_sents()[22]
    tokens = [token for token, _ in sent]
    print(st.tag(tokens=tokens))


if __name__ == '__main__':
    main()
