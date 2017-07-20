Frame-semantic parser for automatically detecting semantic frames and their arguments, with the respective labels, from text. Uses a softmax-margin segmental RNN model.

This README only contains instructions for running the baseline SegRNN model, instructions for other models will be updated soon.

#### Required software

 * A C++ compiler supporting the [C++11 language standard](https://en.wikipedia.org/wiki/C%2B%2B11)
 * [Boost](http://www.boost.org/) libraries
 * [Eigen](http://eigen.tuxfamily.org) (newer versions strongly recommended)
 * [CMake](http://www.cmake.org/)
 * This runs with the Python wrapper to `DyNet-v2`.

#### Linking with DyNet
Create a dynet-base/ directory here. Follow the [python installation directions](http://dynet.readthedocs.io/en/latest/python.html) to set up DyNet.

#### Preprocessing FrameNet XML files

First, create a data/ directory here, download FrameNet version 1.x and place it under data/fndata-1.x/. Also create a directory data/neural/fn1.x/ for the data files in CoNLL 2009 format, which will be generated in the steps below. Now, create the train, test and dev splits from the data by rewriting the xml files provided in CoNLL 2009 format, with BIOS tags, for simplicity, by executing the following:

```python
cd src/
python preprocess.py 2> err

```
The above script writes the train, dev and test files in the required format into the data/ folder.

There is plenty of noise in the data, and all the sentences which could not be converted. The location of these, along with the error is written to the standard error.

To use pretrained GloVe word embeddings, download the [GloVe files](https://nlp.stanford.edu/projects/glove/) and place them under data/. Run the preprocessing with an extra argument for the intended GloVe file. This trims the GloVe files to the FrameNet vocabulary, to ease memory requirements. For example, the command

```python
python preprocess.py glove.6B.100d.txt 2> err

``` 
creates glove.6B.100d.framevocab.txt under data/. 

### Frame Identification
To run the biLSTM frame identification module, execute:

```python
cd src/
python frameid.py \
  --mode test \


```



#### Arg Identification

To run the vanilla SegRNN model, use
```python
cd src/
python segrnn-argid.py 2> err

```

#### Contact

For questions and usage issues, please contact swabha@cs.cmu.edu

