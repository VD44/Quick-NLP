# Quick-NLP
A collection of NLP models and training scripts

Work in progress.

## Installation
Just clone the repository and install the dependencies:

```bash
$ git clone https://github.com/VD44/Quick-NLP

$ cd Quick-Nlp-master

$ pip install requirements.txt
```
## Models

### MLSTM Language Model (Baseline)

A recurrent baseline model for comparison to feedforward models. MLSTM is proposed in the paper: [Multiplicative LSTM for sequence modelling](https://arxiv.org/pdf/1609.07959.pdf). MLSTM is able to use different recurrent transition functions for every possible input, allowing it to be more expressive for autoregressive sequence modeling. MLSTM outperforms standard LSTM and even its deep variants.

Model defined under [Quick-NLP/models/mlstm_lm.py](https://github.com/VD44/Quick-NLP/blob/master/models/mlstm_lm.py).

Recommended to train on wikitext-103 dataset:
```bash
$ bash get_data.sh wikitext

$ python train_mlstm_lm.py
```

### Transformer Decoder Language Model

The decoder component of the architecture described in [Attention Is All You Need](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf). The same model was used to generate wikipedia articles in [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/pdf/1801.10198.pdf). Can be applied to practically any language modeling task. Here we use it as a comparison to the MLSTM language model as well as to the task of question generation.

A single block of the model is depicted below:

### Transformer Entailment

### QANet (Reading Comprehension)

## Datasets

## Training

## Saved Params

## Papers:

* [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
* [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
* [QANet: Combining Local Convolution With Global Self-Attention For Reading Comprehension](https://openreview.net/pdf?id=B14TlG-RW)
* [Learning to Generate Reviews and Discovering Sentiment](https://arxiv.org/pdf/1704.01444.pdf)
* [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/pdf/1801.10198.pdf)
