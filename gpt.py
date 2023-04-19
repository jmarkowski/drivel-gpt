#!/usr/bin/env python
import torch
from gpt import BigramLanguageModel


RANDOM_SEED = 1337
DEVICE = 'cpu' # 'cpu' or 'mps' (M1 mac specific)


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


def generate_text(tokenizer, model, num_tokens):
    # Create a starting tensor, that we use to "kick off" the generation.
    # 1,1 corresponds to the newline character, which is a reasonable start(?)
    batch = 1
    time = 1
    idx = torch.zeros((batch, time), dtype=torch.long, device=DEVICE)

    model_output = model.generate(idx, max_new_tokens=num_tokens)[0].tolist()

    return tokenizer.decode(model_output)


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

    batch_size = 32 # how many independent sequences will we process in parallel?
    block_size = 64 # what is the maximum context length for predictions?

    # Deterministic randomness
    torch.manual_seed(RANDOM_SEED)

    # A kind of "data loader" to get batch of data
    def get_batch(split):
        """
        Generate a small batch of data of inputs x and targets y
        """
        data = training_data if split == 'train' else validation_data

        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])

        # Move the data to the DEVICE
        x, y = x.to(DEVICE), y.to(DEVICE)

        return x, y

    xb, yb = get_batch('train')

    # Feed the tensor data into a neural network. The Bigram Language model is
    # the simplest neural network.
    model = BigramLanguageModel(vocab_size, device=DEVICE)
    m = model.to(DEVICE) # move model parameters to DEVICE
    logits, loss = model(xb, yb)

    # Let's create our first generation!!! It looks like garbage though,
    # because our model is totally random (it hasn't been trained!).
    #
    # Generated text:
    # lfJeukRuaRJKXAYtXzfJ:HEPiu--sDioi;ILCo3pHNTmDwJsfheKRxZCFs
    # lZJ XQc?:s:HEzEnXalEPklcPU cL'DpdLCafBheH
    print(f'Generated text: {generate_text(t, m, 100)}')

    # Create a PyTorch optimizer
    # Available options include, for example SGD (stochastic gradient descent)
    # or AdamW, which is a more popular and advanced optimizer that works
    # extremely well.
    learning_rate = 1e-3 # typically 3e-4 is good, but for small neural networks
                         # we can use much higher learning rates.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    eval_iters = 200
    eval_interval = 100

    # improve memory efficiency by telling it we will not be going backwards
    @torch.no_grad()
    def estimate_loss():
        out = {}

        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logs, loss = model(X, Y)
                losses[k] = loss.item()

            out[split] = losses.mean()

        model.train()

        return out

    # Typical training loop ...
    batch_size = 32
    steps = 1000 # Increase for good results ... Increasing it reduces the loss.
    for s in range(steps):
        if s % eval_interval == 0:
            losses = estimate_loss()
            t_loss = losses['train']
            v_loss = losses['val']

            print(f'Step {s}: training loss {t_loss:.4f}, validation loss {v_loss:.4f}');

        # Sample a batch of data
        xb, yb = get_batch('train')

        # Evaluate the loss
        _logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f'Loss from running optimizer: {loss.item()}')

    # With steps=100000, we get something closer to our data...
    #
    # Generated text:
    # Wigauther LLIZARI gatho ftcohanghorad
    # Age cur, aur hayis;
    # Wheano?
    # QUpe.
    # N otord, fane hiler, withy f
    print(f'Generated text: {generate_text(t, m, 500)}')


if __name__ == '__main__':
    main()
