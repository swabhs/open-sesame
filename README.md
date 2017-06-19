Frame-semantic parser for automatically detecting semantic frames and their arguments, with the respective labels, from text. Uses a softmax-margin segmental RNN model.

#### Required software

 * A C++ compiler supporting the [C++11 language standard](https://en.wikipedia.org/wiki/C%2B%2B11)
 * [Boost](http://www.boost.org/) libraries
 * [Eigen](http://eigen.tuxfamily.org) (newer versions strongly recommended)
 * [CMake](http://www.cmake.org/)
 * This runs with `DyNet-v2`

#### Linking with DyNet
Create a dynet-base/ directory here. Follow the [python installation directions](http://dynet.readthedocs.io/en/latest/python.html) to set up DyNet.

#### Preprocessing FrameNet XML files

First, create a data/ directory here, download FrameNet 1.x and place it under data/ Now, create the train, test and dev splits from the data by rewriting the xml files provided in CoNLL 2009 format, with BIOS tags, for simplicity, by executing the following:

```python
cd src/
python preprocess.py 2> err

```
The above script writes the train, dev and test files in the required format into the data/ folder.

There is plenty of noise in the data, and all the sentences which could not be converted. The location of these, along with the error is written to the standard error.

#### Arg Identification

This step requires DyNet.

```python
cd src/
python segrnn-argid.py 2> err

```

#### Contact

For questions and usage issues, please contact swabha@cs.cmu.edu

