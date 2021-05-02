from nltk.tag import StanfordNERTagger
from nls_cw2.paths import MODEL_7
import nltk


def main():
    # stanford ner tagger.
    st = StanfordNERTagger(MODEL_7)
    sent = nltk.corpus.treebank.tagged_sents()[22]
    tokens = [token for token, _ in sent]
    print(st.tag(tokens=tokens))


if __name__ == '__main__':
    main()
