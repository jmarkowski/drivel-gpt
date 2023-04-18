#!/usr/bin/env python
import torch


def read_text_data(source):
    with open(source, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f'Length of dataset in characters: {len(text)}')

    return text


def get_text_stats(text):
    chars = sorted(set(text))
    vocab_size = len(chars)

    print(f'Chars ({vocab_size}): {"".join(chars)}')

    return chars, vocab_size


class Tokenizer():
    """
    This basic tokenizer is used to support encoding a string of text into an
    array of integers, and vice versa through decoding.
    """
    def __init__(self, char_set):
        self.chars = char_set

    def encode(self, str):
        str2int_map = {c:index for index,c in enumerate(self.chars)}
        encoded_vals = [str2int_map[c] for c in str]

        #print(f'Encoded "{str}" to {encoded_vals}')

        return encoded_vals

    def decode(self, input_vals):
        int2str_map = {index:c for index,c in enumerate(self.chars)}
        decoded_str = ''.join([int2str_map[i] for i in input_vals])

        #print(f'Decoded {input_vals} to "{decoded_str}"')

        return decoded_str


def get_data_tensor(tokenizer, text):
    """
    Encode the entire text dataset and store it in a tensor.
    """
    data_tensor = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    return data_tensor


def main():
    input = 'input.txt'

    text = read_text_data(input)

    chars, vocab_size = get_text_stats(text)

    # Create tokenizer
    t = Tokenizer(chars)
    t.encode('hello world')
    t.decode(t.encode('hello world'))

    data_tensor = get_data_tensor(t, text)

    n = int(0.9*len(data_tensor))
    training_data = data_tensor[:n]
    validation_data = data_tensor[n:]

    block_size = 8
    print(training_data[:block_size+1])


if __name__ == '__main__':
    main()
