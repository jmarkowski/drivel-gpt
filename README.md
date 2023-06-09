# Drivel GPT

This codebase contains a simple, custom version of a language model using a
Generative Pretrained Transformer (GPT) model, created by following along
Andrej Karpathy's
[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY),
which models its implementation after the landmark paper on transformers, namely
[Attention is All You Need](https://arxiv.org/abs/1706.03762).

The output generated by this GPT model will depend on the dataset that it's
trained on, and often times, will just be gibberish. Hence, the name :)

## Purpose

To understand and appreciate how a GPT model works under the hood.

## Sample Output

```
QUEEN:
Thou wranst I am dear with'd, in like a speak;
The dring wife-hear'd unstrard, whet hereing with
where broud them; menny our wringled canspess;
for here with like her cheep? Gremion, my solurand,
Good lord, a truip gateny and you.

DUKE VINCENTIO:
But by oyid miney rey, was that I no, 'That with lown'd
Deling I disacious nursed spakes on nurbland sts.

POMPEYON:
And, by The warm, lord! darchs'-moth and i

```

## Installation

### Install Conda

    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

    sh Miniconda3-latest-MacOSX-arm64.sh

### Install pytorch

    conda install pytorch=1.13.1 -c pytorch

## Resources

### Videos

* [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)
* [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)

### Papers

* [Attention is All You Need](https://arxiv.org/abs/1706.03762)
* [OpenAI's GPT-3 Paper](https://arxiv.org/pdf/2005.14165.pdf)

### Repositories

* [nanoGPT](https://github.com/karpathy/nanoGPT)

### Articles & Media

* [The Illistrated GPT-2](http://jalammar.github.io/illustrated-gpt2/)
* [GPT Visual Guide](https://twitter.com/akshay_pachaar/status/1647940492712345601)
* [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)
