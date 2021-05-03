import nltk


def main():
    str_format = "(S\n  The/DT\n  river/NN\n  has/VBZ\n  not/RB\n  only/RB\n  become/VB\n  the/DT\n  property/NN\n  of/IN\n  the/DT\n  (GPE United/NNP States/NNPS)\n  from/IN\n  its/PRP$\n  source/NN\n  to/TO\n  the/DT\n  ocean/NN\n  ,/,\n  with/IN\n  all/DT\n  its/PRP$\n  tributary/JJ\n  streams/NNS\n  (/(\n  with/IN\n  the/DT\n  exception/NN\n  of/IN\n  the/DT\n  upper/JJ\n  part/NN\n  of/IN\n  the/DT\n  (ORGANIZATION Red/NNP)\n  River/NNP\n  only/RB\n  ),/VBZ\n  but/CC\n  (PERSON Louisiana/NNP)\n  ,/,\n  with/IN\n  a/DT\n  fair/JJ\n  and/CC\n  liberal/JJ\n  boundary/NN\n  on/IN\n  the/DT\n  western/JJ\n  side/NN\n  and/CC\n  the/DT\n  (ORGANIZATION Floridas/NNP)\n  on/IN\n  the/DT\n  eastern/JJ\n  ,/,\n  have/VBP\n  been/VBN\n  ceded/VBN\n  to/TO\n  us/PRP\n  ./.))"
    tree = nltk.Tree.fromstring(str_format)
    print(tree)


if __name__ == '__main__':
    main()