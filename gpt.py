#!/usr/bin/env python
def read_text_data(source):
    with open(source, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f'Length of dataset in characters: {len(text)}')

    return text


def main():
    input = 'input.txt'

    text = read_text_data(input)


if __name__ == '__main__':
    main()
