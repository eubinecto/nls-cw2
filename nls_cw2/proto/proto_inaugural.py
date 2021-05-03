
import nltk
from nltk.corpus import inaugural


def main():
    nltk.download('inaugural')
    for sent in inaugural.sents():
        print(sent)


if __name__ == '__main__':
    main()
