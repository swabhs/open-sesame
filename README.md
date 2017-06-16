This is a frame-semantic parser for automatically detecting semantic frames and their arguments, with the respective labels, from text.


# Preprocessing FrameNet XML files

First, create a data/ directory here, download FrameNet 1.x and place it under data/ Now, create the train, test and dev splits from the data by rewriting the xml files provided in CoNLL 2009 format, with BIOS tags, for simplicity.

```python
cd src/
python preprocess.py 2> err

```
The above script writes the train, dev and test files in the required format into the data/ folder.

There is a plenty of noise in the data, and all the sentences which could not be converted. The location of these, along with the error is written to the standard error.

# Target / LU Identfication

This step is skipped in our pipeline. Please read Das et. al. (CL 2013) for details.

# Frame Identification

This step requires DyNet.

```python
python identification.py
```

# Arg Identification

This step also requires DyNet.

## Basic SegRNN model

## Dependency/Constituency baseline

## Scaffolding model
