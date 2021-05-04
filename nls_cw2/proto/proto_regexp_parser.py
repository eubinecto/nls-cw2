import nltk


def main():
    sent_1 = "one of the greatest family-oriented , fantasy-adventure movies ever ."
    sent_2 = "a thoughtful , provocative , inspiring and insistently humanizing film ."
    sent_3 = "guaranteed to move anyone who ever shook , rattled , or rolled ."
    tokens_1 = nltk.word_tokenize(sent_1)
    tokens_2 = nltk.word_tokenize(sent_2)
    tokens_3 = nltk.word_tokenize(sent_3)
    tree_1 = nltk.pos_tag(tokens_1)
    tree_2 = nltk.pos_tag(tokens_2)
    tree_3 = nltk.pos_tag(tokens_3)
    chunk_rule = """
    ENUM: {<JJ><,><JJ><,><JJ><,><JJ>}
          {<JJ><,><JJ><,><JJ>}
          {<JJ><,><JJ>}  
    """
    parser = nltk.RegexpParser(chunk_rule)
    print(parser.parse(tree_1))
    print(parser.parse(tree_2))
    print(parser.parse(tree_3))


if __name__ == '__main__':
    main()
