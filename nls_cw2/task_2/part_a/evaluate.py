
from nls_cw2.loaders import load_labels


def main():
    labels = load_labels()

    # get the accuracies.
    correct = 0
    total = 0
    for adj, predict, label in labels:
        if predict == label:
            correct += 1
        total += 1

    print("correct:", correct)
    print("total:", total)
    print("accuracy:", correct / total)


if __name__ == '__main__':
    main()
