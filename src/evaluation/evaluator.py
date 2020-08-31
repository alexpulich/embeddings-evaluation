import logging

import numpy as np
import scipy.stats

from sklearn.utils import Bunch

from .strategies.base import StructuredSourceStrategy, OOVStrategy, NoStructuredSourceStrategy
from embeddings.embeddings import Embedding
from config import STRUCTURED_SOURCE_COEF_RANGE

logger = logging.getLogger(__name__)


class ThaiEvaluator:
    def __init__(self,
                 oov_strategy: OOVStrategy,
                 structured_source_strategy: StructuredSourceStrategy
                 ):
        self._structured_source_strategy = structured_source_strategy
        self._oov_strategy = oov_strategy

    @property
    def structured_source_strategy(self) -> StructuredSourceStrategy:
        return self._structured_source_strategy

    @structured_source_strategy.setter
    def structured_source_strategy(self, strategy: StructuredSourceStrategy) -> None:
        self._structured_source_strategy = strategy

    @property
    def oov_strategy(self) -> OOVStrategy:
        return self._oov_strategy

    @oov_strategy.setter
    def oov_strategy(self, strategy: OOVStrategy) -> None:
        self._oov_strategy = strategy

    def _evaluate(self,
                  embeddings: Embedding,
                  X,
                  y,
                  filter_not_found: bool = False,
                  structured_source_coef=None
                  ):

        if isinstance(embeddings, dict):
            embeddings = Embedding.from_dict(embeddings)

        missing_words, found_words, oov_vecs_created, index = 0, 0, 0, 0
        word_pair_oov_indices = []

        words = embeddings.vocabulary.word_id

        # applying oov handling strategy if any
        self._oov_strategy.handle_oov(embeddings, X, words)

        ## For all words in the datasets, check if the are OOV?
        ## Indices of word-pairs with a OOV word are stored in word_pair_oov_indices
        for query in X:
            for query_word in query:

                if query_word not in words:
                    logger.debug("Missing Word:", query_word)
                    missing_words += 1
                    word_pair_oov_indices.append(index)
                else:
                    logger.debug("Found Word:", query_word)
                    found_words += 1
            index += 1

        word_pair_oov_indices = list(set(word_pair_oov_indices))
        logger.debug('word_pair_oov_indices', word_pair_oov_indices)

        if missing_words > 0 or oov_vecs_created > 0:
            logger.warning("Missing {} words. Will replace them with mean vector".format(missing_words))
            logger.warning(
                "OOV words {} created from their subwords. Will replace them with mean vector of sub-tokens".format(
                    oov_vecs_created))
            logger.warning("Found {} words.".format(found_words))

        logger.debug('X.shape', X.shape)
        logger.debug('y.shape', y.shape)

        if filter_not_found:
            new_X = np.delete(X, word_pair_oov_indices, 0)
            new_y = np.delete(y, word_pair_oov_indices)

            logger.debug('new_X.shape', new_X.shape)
            logger.debug('new_y.shape', new_y.shape)

            mean_vector = np.mean(embeddings.vectors, axis=0, keepdims=True)
            A = np.vstack(embeddings.get(word, mean_vector) for word in new_X[:, 0])
            B = np.vstack(embeddings.get(word, mean_vector) for word in new_X[:, 1])
            logger.debug(len(A), len(B))
            logger.debug(type(A), type(B))
            scores = np.array([v1.dot(v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2)) for v1, v2 in zip(A, B)])

            y = new_y
            pairs = new_X

        else:
            mean_vector = np.mean(embeddings.vectors, axis=0, keepdims=True)

            A = np.vstack([embeddings.get(word, mean_vector) for word in X[:, 0]])
            B = np.vstack([embeddings.get(word, mean_vector) for word in X[:, 1]])
            scores = np.array([v1.dot(v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
            pairs = X

        self._oov_strategy.handle_oov(embeddings, X, words)

        scores = self._structured_source_strategy.apply(scores, pairs, structured_source_coef)

        # taking ranks for model (scores) and dataset (y)
        scores_rank = scipy.stats.rankdata(scores).tolist()
        y_rank = scipy.stats.rankdata(y).tolist()

        pairs_list = pairs.tolist()
        scores_rank = {tuple(pair): rank for pair, rank in zip(pairs_list, scores_rank)}
        y_rank = {tuple(pair): rank for pair, rank in zip(pairs_list, y_rank)}

        result = {'spearmanr': scipy.stats.spearmanr(scores, y).correlation,
                  'pearsonr': scipy.stats.pearsonr(scores, y)[0],
                  'num_oov_word_pairs': len(word_pair_oov_indices),
                  'num_found_words': found_words,
                  'num_missing_words': missing_words,
                  'num_oov_created': oov_vecs_created,
                  'y.shape': y.shape,
                  'scores_rank': scores_rank,
                  'y_rank': y_rank
                  }

        # TODO
        # if not isinstance(self.structured_source_strategy, NoStructuredSourceStrategy):
        #     result['structed_oov_pairs'] = structed_oov_pairs

        return result

    def evaluate(self,
                 embeddings: Embedding,
                 dataset_data: Bunch,
                 filter_not_found: bool = False
                 ):

        X = dataset_data.X
        y = dataset_data.y

        if isinstance(self.structured_source_strategy, NoStructuredSourceStrategy):
            result = self._evaluate(embeddings, X, y, filter_not_found)
            try:
                hm = scipy.stats.hmean([result['spearmanr'], result['pearsonr']])
            except:
                hm = -999  ## undefined
            return result, hm
        else:
            results = []
            for coef in STRUCTURED_SOURCE_COEF_RANGE:
                result = self._evaluate(embeddings, X, y, filter_not_found, coef)
                result['coef'] = coef
                try:
                    result['hm'] = scipy.stats.hmean([result['spearmanr'], result['pearsonr']])
                except:
                    result['hm'] = -999  ## undefined
                results.append(result)
            result = max(results, key=lambda x: x['hm'])
            logger.debug('BEST COEF: {}'.format(result['coef']))

            # TODO
            # logger.debug('STRUCTED OOV : {}'.format(result['structed_oov_pairs']))
            return result
