Frame-semantic parser for automatically detecting FrameNet frames and their frame-elements from sentences. Uses a softmax-margin segmental recurrent neural net model, based on the paper [Frame-Semantic Parsing with Softmax-Margin Segmental RNNs and a Syntactic Scaffold](https://arxiv.org/abs/1706.09528).

This `README` only contains instructions for running the baseline model from [our paper](https://arxiv.org/abs/1706.09528), instructions for other models will be updated soon.

## Installation

This project is developed using Python 2.7. Other requirements include the [DyNet](http://dynet.readthedocs.io/en/latest/python.html) library, and some [NLTK](https://www.nltk.org/) packages.

```sh
pip install dynet
pip install nltk
python -m nltk.downloader averaged_perceptron_tagger wordnet
```

## Data Preprocessing

This codebase only handles data in the XML format specified under FrameNet. However, we convert the data into [CoNLL 2009 format](https://ufal.mff.cuni.cz/conll2009-st/task-description.html) for ease of use.

First, create a `data/` directory here, download [FrameNet](https://framenet.icsi.berkeley.edu/fndrupal/framenet_data) version 1.x and place it under `data/fndata-1.x/`. Also create a directory `data/neural/fn1.x/` for the data files in , which will be generated in the steps below. Now, create the train, test and dev splits from the data by rewriting the xml files provided in CoNLL 2009 format, with BIOS tags, for simplicity, by executing the following:

```sh
cd src/
python preprocess.py 2> err
```
The above script writes the train, dev and test files in the required format into the `data/` directory.

There is plenty of noise in the data, and all the sentences which could not be converted. The location of these, along with the error is written to the standard error.

To use pretrained GloVe word embeddings, download the [GloVe files](https://nlp.stanford.edu/projects/glove/) and place them under `data/`. Run the preprocessing with an extra argument for the intended GloVe file. This trims the GloVe files to the FrameNet vocabulary, to ease memory requirements. For example, the command

```sh
python preprocess.py glove.6B.100d.txt 2> err
``` 
creates `glove.6B.100d.framevocab.txt` under `data/`.

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
python frameid.py --mode test > frameid.log
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

