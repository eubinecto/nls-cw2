import nltk


def main():
    # need to download all of these
    nltk.download('treebank')  # to get an example sentence
    nltk.download('maxent_ne_chunker')  # the trained ner classifier
    nltk.download('words')  # somehow need this as well
    # this is a tagged sentence
    sent = nltk.corpus.treebank.tagged_sents()[22]
    nes = nltk.ne_chunk(tagged_tokens=sent)  # e.g. PERSON, ORGANIZATION, and GPE.
    print(nes)
    print("---")


if __name__ == '__main__':
    main()
