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

2. Convert the data into a [format similar to CoNLL 2009](https://ufal.mff.cuni.cz/conll2009-st/task-description.html), but with BIO tags, by executing:
```sh
cd src/
python preprocess.py 2> err
```
The above script writes the train, dev and test files in the required format into the `data/neural/fn1.x/` directory. There is plenty of noise in the annotations. The annotations which could not be used, along with the error messages, gets spit out to the standard error.

3. [Optional, but highly recommended] If you want to use pretrained [GloVe word embeddings](https://nlp.stanford.edu/projects/glove/), download and extract them under `data/`. Run the preprocessing with an extra argument for the intended GloVe file.

```sh
python preprocess.py glove.6B.100d.txt 2> err
``` 
This trims the GloVe files to the FrameNet vocabulary, to ease memory requirements. For example, the above creates `data/glove.6B.100d.framevocab.txt` to be used by our models.

## Target Identification

A bidirectional LSTM model takes into account the lexical unit index in FrameNet to identify targets. This model is *not* described in the [paper](https://arxiv.org/abs/1706.09528).

#### Training
To train the target identification module, execute:

```sh
cd src/
python targetid.py
```
This saves the best model on validation data in the directory `src/tmp/`, which will be pointed to by the symbolic link `src/model.targetid.1.x`. Pre-trained models coming soon.

#### Test
To test under the best model in `src/model.targetid.1.x`, execute:

```sh
python targetid.py --mode test
```

## Frame Identification

Frame identification is based on a bidirectional LSTM model.

#### Training
To train the frame identification module, execute:

```sh
cd src/
python frameid.py
```
This saves the best model on validation data in the directory `src/tmp/`, which will be pointed to by the symbolic link `src/model.frameid.1.x`. Pre-trained models coming soon.

#### Test
To test under the best model in `src/model.frameid.1.x`, execute:

```sh
python frameid.py --mode test > frameid.log
```
`frameid.log` will contain example-wise analysis. The output, in CoNLL 2009 format will be written to `predicted.1.x.frameid.test.out` and in the [frame-elements file format](https://github.com/Noahs-ARK/semafor/tree/master/training/data) to `my.predict.test.frame.elements`.

## Frame-Element (Argument) Identification

Argument identification is based on a segmental recurrent neural net model, used as a baseline in [our paper](https://arxiv.org/abs/1706.09528).

#### Training
To train an argument identifier, execute:
```sh
cd src/
python segrnn-argid.py 2> err
```
This saves the best model on validation data in the directory `src/tmp/`, which will be pointed to by the symbolic link `src/model.segrnn-argid.1.x`. Pre-trained models coming soon.

#### Test
To test under the best model in `src/model.segrnn-argid.1.x`, execute:

```sh
python segrnn-argid.py --mode test > argid.log
```

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

