from nls_cw2.loaders import load_mpqa_lexicons


def main():
    for lemma, senti in load_mpqa_lexicons().items():
        print(lemma, senti)


if __name__ == '__main__':
    main()