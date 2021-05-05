import nltk


def main():
    pass
    sent_1 = "one of the greatest family-oriented , fantasy-adventure movies ever ."
    sent_2 = "a thoughtful , provocative and insistently humanizing film ."
    sent_3 = "guaranteed to move anyone who ever shook , rattled , or rolled ."
    tagged_1 = nltk.pos_tag(sent_1.split(" "))
    tagged_2 = nltk.pos_tag(sent_2.split(" "))
    tagged_3 = nltk.pos_tag(sent_3.split(" "))
    tree_1 = nltk.ne_chunk(tagged_1)
    tree_2 = nltk.ne_chunk(tagged_2)
    tree_3 = nltk.ne_chunk(tagged_3)
    print(tree_1)
    print(tree_2)
    print(tree_3)


if __name__ == '__main__':
    main()