import os.path as op
import numpy as np

DATASETS_DOWNLOAD_DIR = op.join(op.dirname(op.realpath(__file__)), '../datasets/')


WORDNET_PATH_SIMILARITY_TYPE = 'most_similar'
# WORDNET_PATH_SIMILARITY_TYPE = 'first_synset'

CONCEPTNET_PICKLE_FILE = '../conceptnet.pickle'

NUMBERBATCH_PATH = '/Users/alexpulich/wrk/uni/nlp/numberbatch_wo_prefix_fin.txt'

STRUCTURED_SOURCE_COEF_RANGE = np.arange(0.00, 1.1, 0.1)