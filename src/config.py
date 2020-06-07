import logging
import os.path as op
import numpy as np

PROJECT_DIR = op.join(op.dirname(op.realpath(__file__)), '../')

DATASETS_DOWNLOAD_DIR = op.join(PROJECT_DIR, 'datasets/')


WORDNET_PATH_SIMILARITY_TYPE = 'most_similar'
# WORDNET_PATH_SIMILARITY_TYPE = 'first_synset'

CONCEPTNET_PICKLE_FILE = op.join(PROJECT_DIR, 'models/conceptnet.pickle')

NUMBERBATCH_PATH = op.join(PROJECT_DIR, 'models/numberbatch_wo_prefix_fin.txt')

STRUCTURED_SOURCE_COEF_RANGE = np.arange(0.00, 1.1, 0.1)

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')