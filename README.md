Frame-semantic parser for automatically detecting [FrameNet](https://framenet.icsi.berkeley.edu/fndrupal/) frames and their frame-elements from sentences. The model is based on  softmax-margin segmental recurrent neural nets, described in our paper [Frame-Semantic Parsing with Softmax-Margin Segmental RNNs and a Syntactic Scaffold](https://arxiv.org/abs/1706.09528). An example of a frame-semantic parse is shown below

![Frame-semantics example](fig/fsp-example.png)

## Installation

This project is developed using Python 2.7. Other requirements include the [DyNet](http://dynet.readthedocs.io/en/latest/python.html) library, and some [NLTK](https://www.nltk.org/) packages.

```sh
pip install dynet
pip install nltk
python -m nltk.downloader averaged_perceptron_tagger wordnet
```

## Data Preprocessing

This codebase only handles data in the XML format specified under FrameNet. However, we first reformat the data for ease of readability.

1. First, create a `data/` directory here, download FrameNet version 1.x and place it under `data/fndata-1.x/`. Also create a directory `data/neural/fn1.x/` to convert to CoNLL 2009 format.

2. Second, this project uses pretrained [GloVe word embeddings](https://nlp.stanford.edu/projects/glove/). Download and extract them under `data/`.

2. Convert the data into a [format similar to CoNLL 2009](https://ufal.mff.cuni.cz/conll2009-st/task-description.html), but with BIO tags, by executing:
```sh
python -m src.preprocess glove.6B.100d.txt 2> err
```
The above script writes the train, dev and test files in the required format into the `data/neural/fn1.x/` directory. There is plenty of noise in the annotations. The annotations which could not be used, along with the error messages, gets spit out to the standard error. Also trims the GloVe files to the FrameNet vocabulary, to ease memory requirements. For example, the above creates `data/glove.6B.100d.framevocab.txt` to be used by our models.


## Training

Here, we briefly describe the training for each module. The different modules are target identification, frame identification and argument identification, which *need to be executed in that order*. To train a module, execute:

```sh
python -m src.`MODULE` --model_name sample-model --mode train
```

The `MODULE`s are specified below. Training saves the best model on validation data in the directory `logs/sample-model/best-MODULE-1.x-model`. Pre-trained models coming soon.

If training gets interrupted, you can restart from the last saved checkpoint by specifying `--mode refresh`.

## Test
To test under the above model, execute:

```sh
python -m src.`MODULE` --model_name sample-model --mode test
```

The output, in a CoNLL 2009-like format will be written to `logs/sample-model/predicted-1.x-MODULE-test.conll` and in the [frame-elements file format](https://github.com/Noahs-ARK/semafor/tree/master/training/data) to `logs/sample-model/predicted-test.fes` for frame and argument identification.

### 1. Target Identification

`MODULE=targetid`

A bidirectional LSTM model takes into account the lexical unit index in FrameNet to identify targets. This model has *not* been described in the [paper](https://arxiv.org/abs/1706.09528).

### 2. Frame Identification

`MODULE=frameid`

Frame identification is based on a bidirectional LSTM model. Targets and their respective lexical units need to be identified before this step. 
At test time, the module spits out `frameid.log` containing example-wise analysis.

### 3. Argument (Frame-Element) Identification

`MODULE=segrnn-argid`

Argument identification is based on a segmental recurrent neural net, used as the *baseline* in [our paper](https://arxiv.org/abs/1706.09528). Targets and their respective lexical units need to be identified, and frames corresponding to the LUs predicted before this step.

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

