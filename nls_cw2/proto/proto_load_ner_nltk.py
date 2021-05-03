from nls_cw2.loaders import load_ner_with_nltk


def main():
    for idx, chunks in enumerate(load_ner_with_nltk()):
        print(idx)
        print(chunks)


if __name__ == '__main__':
    main()