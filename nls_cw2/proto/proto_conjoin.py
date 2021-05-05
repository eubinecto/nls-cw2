import nltk


def main():
    sent = "a thoughtful , provocative and inspiring film ."
    tokens = nltk.word_tokenize(sent)
    tree = nltk.pos_tag(tokens)
    chunk_rule = """
    CONJOIN: {<JJ><CC><JJ>}
    """
    parser = nltk.RegexpParser(chunk_rule)
    print(parser.parse(tree))


if __name__ == '__main__':
    main()
