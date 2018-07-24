Frame-semantic parser for automatically detecting [FrameNet](https://framenet.icsi.berkeley.edu/fndrupal/) frames and their frame-elements from sentences. The model is based on  softmax-margin segmental recurrent neural nets, described in our paper [Frame-Semantic Parsing with Softmax-Margin Segmental RNNs and a Syntactic Scaffold](https://arxiv.org/abs/1706.09528). An example of a frame-semantic parse is shown below

![Frame-semantics example](fig/fsp-example.png)

## Installation

This project is developed using Python 2.7. Other requirements include the [DyNet](http://dynet.readthedocs.io/en/latest/python.html) library, and some [NLTK](https://www.nltk.org/) packages.

```sh
$ pip install dynet
$ pip install nltk
$ python -m nltk.downloader averaged_perceptron_tagger wordnet
```

## Data Preprocessing

This codebase only handles data in the XML format specified under FrameNet. However, we first reformat the data for ease of readability.

1. First, create a `data/` directory here, download FrameNet version 1.$x and place it under `data/fndata-1.$x/`. Also create a directory `data/neural/fn1.$x/` to convert to CoNLL 2009 format.

2. Second, this project uses pretrained [GloVe word embeddings](https://nlp.stanford.edu/projects/glove/) of 100 dimensions. Download and extract them under `data/`.

3. Convert the data into a [format similar to CoNLL 2009](https://ufal.mff.cuni.cz/conll2009-st/task-description.html), but with BIO tags, by executing:
```sh
$ python -m sesame.preprocess glove.6B.100d.txt 2> err
```
The above script writes the train, dev and test files in the required format into the `data/neural/fn1.$x/` directory. There is plenty of noise in the annotations --- annotations which could not be used, along with their respective error messages, get spit out to the standard error. The script also trims the GloVe files to the FrameNet vocabulary, to ease memory requirements. For example, the above creates `data/glove.6B.100d.framevocab.txt` to be used by our models.


## Training

Here, we briefly describe the training for each model. The different models are target identification, frame identification and argument identification, which *need to be executed in that order*. To train a model, execute:

```sh
$ python -m sesame.$MODEL --model_name sample-$MODEL --mode train
```

The $MODELs are specified below. Training saves the best model on validation data in the directory `logs/sample-$MODEL/best-$MODEL-1.$x-model`. The same directory will also save a `configurations.json` containing current model configuration. Pre-trained models coming soon.

If training gets interrupted, it can be restarted from the last saved checkpoint by specifying `--mode refresh`.

## Test
To test under the above model, execute:

```sh
$ python -m sesame.$MODEL --model_name sample-$MODEL --mode test
```

The output, in a CoNLL 2009-like format will be written to `logs/sample-$MODEL/predicted-1.$x-$MODEL-test.conll` and in the [frame-elements file format](https://github.com/Noahs-ARK/semafor/tree/master/training/data) to `logs/sample-$MODEL/predicted-1.$x-$MODEL-test.fes` for frame and argument identification.

### 1. Target Identification

`$MODEL = targetid`

A bidirectional LSTM model takes into account the lexical unit index in FrameNet to identify targets. This model has *not* been described in the [paper](https://arxiv.org/abs/1706.09528).

### 2. Frame Identification

`$MODEL = frameid`

Frame identification is based on a bidirectional LSTM model. Targets and their respective lexical units need to be identified before this step. At test time, example-wise analysis is logged in the model directory.

### 3. Argument (Frame-Element) Identification

`$MODEL = segrnn-argid`

Argument identification is based on a segmental recurrent neural net, used as the *baseline* in the [paper](https://arxiv.org/abs/1706.09528). Targets and their respective lexical units need to be identified, and frames corresponding to the LUs predicted before this step. At test time, example-wise analysis is logged in the model directory.

## Prediction on unannotated data

For predicting targets, frames and arguments on unannotated data, pretrained models are needed. Input needs to be specified in a file containing one sentence per line. The following steps result in the full frame-semantic parsing of the sentences:

```sh
$ python sesame.targetid --model_name pretrained-targetid --mode predict --raw_input sentences.txt
$ python sesame.frameid --model_name pretrained-frameid --mode predict --raw_input logs/pretrained-targetid/predicted-targets.conll
$ python sesame.segrnn-argid --model_name pretrained-argid --mode predict --raw_input logs/pretrained-frameid/predicted-frames.conll
```

The resulting frame-semantic parses will be written to `logs/pretrained-argid/predicted-args.conll` in the same CoNLL 2009-like format.

## Contact and Reference

For questions and usage issues, please contact `swabha@cs.cmu.edu`. If you use open-sesame for research, please cite [our paper](https://arxiv.org/pdf/1706.09528.pdf) as follows:

```
@article{swayamdipta:17,
  title={{Frame-Semantic Parsing with Softmax-Margin Segmental RNNs and a Syntactic Scaffold}},
  author={Swabha Swayamdipta and Sam Thomson and Chris Dyer and Noah A. Smith},
  journal={arXiv preprint arXiv:1706.09528},
  year={2017}
}
```