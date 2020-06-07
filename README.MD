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
1. Unzip models/models.zip into the models dir.
2. Check config.py:
 
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

## Usage
```
Usage: run.py [OPTIONS] MODEL FORMAT

  Evaluation tool for Thai distributional models with option of integrating
  structured sources(WordNet, ConceptNet)

Options:
  --oov [deepcut|letters]  Strategy to handle OOV: letters or deepcut. If
                           empty none of them is applied

  --ss [wn1|wn2|cn1|cn2]   Integrating structed sources. If empty only word
                           embeddings are evaluated

  -f                       Filter not found words
  -m, --multiprocess       Using multiprocessing to parallel datasets
                           evaluation

  --help                   Show this message and exit.

```

# Notes
This code is built on the basis of [Kudkudak tool](https://github.com/kudkudak/word-embeddings-benchmarks).
Some code is copied as is

# TODO 
describe license 