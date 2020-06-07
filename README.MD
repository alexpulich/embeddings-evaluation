# Requirements
- python 3.7+
- packages from requirements.txt

# Building
Use some virtual environment (virtualenv, pipenv, etc) and install dependencies.

Example:
```
virtualenv venv
source ./venv/bin/activate 
pip install -r requirements.txt
```

# Running

## Before running
Check config.py:
 
- You can set WordNet strategy (WORDNET_PATH_SIMILARITY_TYPE)
- You should set path to your numberbatch model file (NUMBERBATCH_PATH)
- CONCEPTNET_PICKLE_FILE is required if you are using dict with ConceptNet scores saved from ConceptNet web API with pickle


## Running
```
cd src
python run.py <path_to_model> <model_file_format>
```

## Supported model formats
1. word2vec - word2vec text format
2. word2vec_bin - word2vec binary format 
3. glove - **untested**
4. dict - **untested**

## Optional parameters
```
--oov - strategy to work with oov words
Options:
  deepcut - apply deepcut
  letters - cut letter by letter until some words with such prefix are found in the vocabulary. Then take an average vector for words with such prefix.

--ss - structured sources using
Options:
  wn1 - using wordnet with method 1 from the paper
  wn2 - using wordnet with method 2 from the paper
  cn1 - usign conceptnet with method 1 from the paper
  cn2 - usign conceptnet with method 2 from the paper

-f - ignore not found words
```

# Notes
This code is built on the basis of [Kudkudak tool](https://github.com/kudkudak/word-embeddings-benchmarks).
Some code is copied as is

# TODO 
describe license 