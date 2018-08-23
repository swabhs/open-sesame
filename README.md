# Open-SESAME

A frame-semantic parser for automatically detecting [FrameNet](https://framenet.icsi.berkeley.edu/fndrupal/) frames and their frame-elements from sentences. The model is based on  softmax-margin segmental recurrent neural nets, described in our paper [Frame-Semantic Parsing with Softmax-Margin Segmental RNNs and a Syntactic Scaffold](https://arxiv.org/abs/1706.09528). An example of a frame-semantic parse is shown below

![Frame-semantics example](fig/fsp-example.png)

## Installation

This project is developed using Python 2.7. Other requirements include the [DyNet](http://dynet.readthedocs.io/en/latest/python.html) library, and some [NLTK](https://www.nltk.org/) packages.

```sh
$ pip install dynet
$ pip install nltk
$ python -m nltk.downloader averaged_perceptron_tagger wordnet
```

## Data Preprocessing

This codebase only handles data in the XML format specified under FrameNet. The default version is FrameNet 1.7, but the codebase is backward compatible with versions 1.6 and 1.5.

As a first step the data is preprocessed for ease of readability.

1. First, create a `data/` directory here, download FrameNet version 1.$x and place it under `data/fndata-1.$x/`.

2. Second, this project uses pretrained [GloVe word embeddings](https://nlp.stanford.edu/projects/glove/) of 100 dimensions. Download and extract them under `data/`.

3. Optionally, make alterations to the configurations in `configurations/global_config.json`, if you have decided to either use a different version of FrameNet, or different pretrained embeddings, etc.

4. Preprocess the data by first converting into a [format similar to CoNLL 2009](https://ufal.mff.cuni.cz/conll2009-st/task-description.html), but with BIO tags, by executing:
```sh
$ python -m sesame.preprocess
```
The above script writes the train, dev and test files in the required format into the `data/neural/fn1.$x/` directory. There is plenty of noise in the annotations --- annotations which could not be used, along with their respective error messages, get logged as `preprocess-fn1.$x.log`.


## Training

Here, we briefly describe the training for each model. The different models are target identification, frame identification and argument identification, which *need to be executed in that order*. To train a model, execute:

```sh
$ python -m sesame.$MODEL --mode train --model_name sample-$MODEL
```

The $MODELs are specified below. Training saves the best model on validation data in the directory `logs/sample-$MODEL/best-$MODEL-1.$x-model`. The same directory will also save a `configurations.json` containing current model configuration.

If training gets interrupted, it can be restarted from the last saved checkpoint by specifying `--mode refresh`.

## Pre-trained Models

The downloads need to be placed under the base-directory. On extraction, these will create a `logs/` directory containing pre-trained models for target identification, frame identification using gold targets, and argument identification using gold targets and frames.

|           |  FN 1.5 Dev | FN 1.5 Test | FN 1.5 Pretrained Models                                                                             |  FN 1.7 Dev | FN 1.7 Test | FN 1.7 Pretrained Models                                                                             |
|-----------|------------:|------------:|------------------------------------------------------------------------------------------------------|------------:|------------:|------------------------------------------------------------------------------------------------------|
| Target ID |       80.05 |       73.38 | [Download](https://drive.google.com/file/d/1ytGCk_njS2aLXkeB9P4V5JONdI_9BIsm/view?usp=sharing) |       79.78 |       74.21 | [Download](https://drive.google.com/file/d/1pDagzQup--DPOrb21-dIPydwInTSHkMU/view?usp=sharing) |
| Frame ID  |       89.36 |       86.65 | [Download](https://drive.google.com/file/d/1H9VGTQZeo5XQVLvDIjjDsHn4aO6qepAT/view?usp=sharing)  |       89.66 |       86.49 | [Download](https://drive.google.com/file/d/1K6Nc9d4yRai7a1YUSq3EI2-2rivm2uOi/view?usp=sharing)  |
| Arg ID    |       60.6 |        59.24 | [Download](https://drive.google.com/file/d/1FfqihTBpXfdnRY8pgv20sR2KwL5v0y0F/view?usp=sharing)                                                                                          | 60.94 | 61.23 | [Download](https://drive.google.com/file/d/1aBQH6gKx-50xcKUgoqPGsRhVgc4THYgs/view?usp=sharing)                                                                                         |

## Test
To test under the above model, execute:

```sh
$ python -m sesame.$MODEL --mode test --model_name sample-$MODEL
```

The output, in a CoNLL 2009-like format will be written to `logs/sample-$MODEL/predicted-1.$x-$MODEL-test.conll` and in the [frame-elements file format](https://github.com/Noahs-ARK/semafor/tree/master/training/data) to `logs/sample-$MODEL/predicted-1.$x-$MODEL-test.fes` for frame and argument identification.

### 1. Target Identification

`$MODEL = targetid`

A bidirectional LSTM model takes into account the lexical unit index in FrameNet to identify targets. This model has *not* been described in the [paper](https://arxiv.org/abs/1706.09528).

### 2. Frame Identification

`$MODEL = frameid`

Frame identification is based on a bidirectional LSTM model. Targets and their respective lexical units need to be identified before this step. At test time, example-wise analysis is logged in the model directory.

### 3. Argument (Frame-Element) Identification

`$MODEL = argid`

Argument identification is based on a segmental recurrent neural net, used as the *baseline* in the [paper](https://arxiv.org/abs/1706.09528). Targets and their respective lexical units need to be identified, and frames corresponding to the LUs predicted before this step. At test time, example-wise analysis is logged in the model directory.

## Prediction on unannotated data

For predicting targets, frames and arguments on unannotated data, pretrained models are needed. Input needs to be specified in a file containing one sentence per line. The following steps result in the full frame-semantic parsing of the sentences:

```sh
$ python -m sesame.targetid --mode predict \
                            --model_name pretrained-targetid \
                            --raw_input sentences.txt
$ python -m sesame.frameid --mode predict \
                           --model_name pretrained-frameid \
                           --raw_input logs/pretrained-targetid/predicted-targets.conll
$ python -m sesame.argid --mode predict \
                         --model_name pretrained-argid \
                         --raw_input logs/pretrained-frameid/predicted-frames.conll
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
