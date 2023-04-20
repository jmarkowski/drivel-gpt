#!/usr/bin/env python
import torch
from gpt import BigramLanguageModel


RANDOM_SEED = 1337
DEVICE = 'cpu' # 'cpu' or 'mps' (M1 mac specific)

################################################################################
# Configuration parameters
################################################################################
# How many independent sequences will we process in parallel?
BATCH_SIZE = 4

# Number of characters of context to use for predictions
BLOCK_SIZE = 64

# Increase for good results ... Increasing it reduces the loss.
MAX_TRAINING_ITERATIONS = 10000

EVAL_INTERVAL = 1000
EVAL_ITERATIONS = 200

# Typically 3e-4 is good, but for small neural networks we can use much higher
# learning rates.
LEARNING_RATE = 1e-3

# Number of embedding dimensions (n_embed)
D_MODEL = 32

# HEAD_SIZE = D_HEAD = D_MODEL // NUM_HEADS
NUM_HEADS = 4

# The number of layers of the transformer blocks we will have
NUM_LAYERS = 3

# Proportion of intermediate calculations that are dropped out
DROPOUT_RATE = 0.2
################################################################################

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

    tokenizer = Tokenizer(chars)

    # Encode the entire text dataset and store it in a tensor.
    data_tensor = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = int(0.9*len(data_tensor))
    training_data = data_tensor[:n]
    validation_data = data_tensor[n:]

    # Deterministic randomness
    torch.manual_seed(RANDOM_SEED)

    # A kind of "data loader" to get batch of data
    def get_batch(split):
        """
        Generate a small batch of data of inputs x and targets y
        """
        data = training_data if split == 'train' else validation_data

        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])

        # Move the data to the DEVICE
        x, y = x.to(DEVICE), y.to(DEVICE)

        return x, y

    # Feed the tensor data into a neural network. The Bigram Language model is
    # the simplest neural network.
    model = BigramLanguageModel(
            n_layers=NUM_LAYERS,
            n_heads=NUM_HEADS,
            block_size=BLOCK_SIZE,
            vocab_size=vocab_size,
            n_embed=D_MODEL,
            dropout_amount=DROPOUT_RATE,
            device=DEVICE,
        )

    m = model.to(DEVICE) # move model parameters to DEVICE

    n_params = sum(p.numel() for p in m.parameters())
    print(f'N_params = {n_params}')

    # Create a PyTorch optimizer
    # Available options include, for example SGD (stochastic gradient descent)
    # or AdamW, which is a more popular and advanced optimizer that works
    # extremely well.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # improve memory efficiency by telling it we will not be going backwards
    @torch.no_grad()
    def estimate_loss():
        out = {}

        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(EVAL_ITERATIONS)
            for k in range(EVAL_ITERATIONS):
                X, Y = get_batch(split)
                logs, loss = model(X, Y)
                losses[k] = loss.item()

            out[split] = losses.mean()

        model.train()

        return out

    # Typical training loop ...
    max_iterations = MAX_TRAINING_ITERATIONS
    for s in range(max_iterations):
        if s % EVAL_INTERVAL == 0:
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

    # With max_iterations=10000, we get something closer to our data...
    #
    # Generated text:
    # Wigauther LLIZARI gatho ftcohanghorad
    # Age cur, aur hayis;
    # Wheano?
    # QUpe.
    # N otord, fane hiler, withy f
    print(f'Generated text: {generate_text(tokenizer, m, 500)}')


if __name__ == '__main__':
    main()
