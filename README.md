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

The decoder component of the architecture described in [Attention Is All You Need](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf). The same model was used to generate wikipedia articles in [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/pdf/1801.10198.pdf). Can be applied to practically any language modeling task. This model is often used for the task of machine sumerization, here we use it as a comparison to the MLSTM language model as well as for the task of machine question generation.

<img src="./transformer_decoder.png" width="25%">

A single block of the decoder is depicted, be default the model uses 12. The multihead attention is masked such that at every timestep the model can only attend to values up until that timestep. This maintains the autoregressive property of the model. The authors of [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) found that a model pretrained on a large corpus of text can be easily finetuned to model new tasks. In their experiements they pretrained a model on 8 P600 GPU's for 30 days. After finetuning, their model beat the state of the art for 9 of the 12 studied tasks. We use the pretrained weights they published from their experiments as a starting point for the language model. Their code is available at: https://github.com/openai/finetune-transformer-lm. 

Pretrained weights can be imported using:
```bash
$ bash get_data.sh pretrained_lm
```

### Transformer Entailment
Machine Textual Entailment is the task of labeling a pair of statements as being an entailment, a contradiction, or neutral. In this example we finetune a model initialized with the weights described in the Transformer Language Model section above. This model achieves state of the art results at the time of writing. Trained on the [SNLI 1.0](https://nlp.stanford.edu/projects/snli/) corpus. Train with:
```bash
$ bash get_data.sh pretrained_lm snli

$ python train_transformer_snli.py
```

### QANet (Reading Comprehension)
QANet is a feedforward model for Machine Reading Comprehension that takes advantage of self attention and convolution to achieve state of the art results (at time of writing). It is more accurate and much more effecient than classical recurrent architectures. It is trained on the [Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/). The ensemble model described in [QANet: Combining Local Convolution With Global Self-Attention For Reading Comprehension](https://arxiv.org/pdf/1804.09541.pdf) achieves a higher EM (exact match) score than human performance.

<img src="./qanet.png">

The encoder block to the right is used throughout the model, with varying number of convolutional
layers. The use of convolutions allows for the use of layer dropout, a regularization method commonly used in
ConvNets. The model uses a pretrained word embedding using 300 dimensional GloVe vectors and a 200 dimensional trainable character embedding.

Download pretrained GloVe vectors using (Common Crawl 300 dimensional truncated to first 400k words):
```bash
$ bash get_data.sh glove
```

## Datasets

### SNLI
```bash
$ bash get_data.sh snli
```

The SNLI corpus is a benchmark for evaluating natural language inference models. It contained 570k human-written English sentence pairs manually labeled either entailment, contradiction, or neutral. The transformer classification model is used is applied to this task. It uses the pretrained language model params from https://github.com/openai/finetune-transformer-lm. These weights along with the dataset can be downloaded using the get_data.sh script:

```bash
$ bash get_data.sh snli pretrained_lm
# and then train
$ python train_transformer_snli.py
```




### SQuAD
The [Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/) is a dataset for machine reading comprehension, questions and passages are created from Wikipedia articles, the answer to every question is a segment of text from the reading passage.
```bash
$ bash get_data.sh squad
```
The reading comprehension (QANet) and question generation (Transformer Language Model) models are trained on this dataset.
```bash
$ python train_qanet.py
# or
$ python train_transformer_qa_gen.py
```

### Wikitext
```bash
$ bash get_data.sh wikitext
```

## Pretrained Vectors

### GloVe
```bash
$ bash get_data.sh glove
```

### ELMo
```bash
$ bash get_data.sh elmo
```

### Pretrained Language Model
```bash
$ bash get_data.sh pretrained_lm
```

## Training

## Saved Params

## Papers:

* [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
* [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
* [QANet: Combining Local Convolution With Global Self-Attention For Reading Comprehension](https://arxiv.org/pdf/1804.09541.pdf)
* [Learning to Generate Reviews and Discovering Sentiment](https://arxiv.org/pdf/1704.01444.pdf)
* [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/pdf/1801.10198.pdf)
