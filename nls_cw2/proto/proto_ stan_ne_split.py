

def main():
    sent = "Martin Luther King * * * * * American Dream *"
    tokens = [
        token.strip()
        for token in sent.split("*")
        if not token.isspace() and token
    ]
    print(tokens)


if __name__ == '__main__':
    main()
