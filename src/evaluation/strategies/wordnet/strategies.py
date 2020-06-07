import logging

import numpy as np

from ..base import StructuredSourceStrategy
from .utils import compute_wordnet_path_scores

logger = logging.getLogger(__name__)
print(__name__)


class WordNetMethod1Strategy(StructuredSourceStrategy):
    """
    Method 1 for the combination of WE and WN scores
    Basic idea: have a coefficient to weight the influence of the two components
    If we don't have a path similarity for a word pair, we use the average path_similarity.
    """

    def apply(self, scores, pairs, structed_sources_coef):
        wn_scores, structed_oov_pairs = compute_wordnet_path_scores(pairs)

        wn_mean = np.mean(np.array(
            [wn_score for wn_score in wn_scores if wn_score is not None]
        ))
        logger.debug("wordnet_method1: avg path similarity:", wn_mean)

        new_scores = []
        for index, pair in enumerate(pairs):
            if wn_scores[index] is None:
                path = wn_mean
            else:
                path = wn_scores[index]

            new_scores.append(structed_sources_coef * scores[index] + (1 - structed_sources_coef) * path)

        return np.array(new_scores)


class WordNetMethod2Strategy(StructuredSourceStrategy):
    """
    Method 2 for the combination of WE and WN scores
    Basic idea: have a coefficient to weight the influence of the two components
    If we don't have a path similarity for a word pair, we use only WE similarity
    Here we transform both the list WE-scores and WN-scores to have mean==0, stddev==1
        -- in order to have them on the same scale when combining them
    """

    def apply(self, scores, pairs, structed_sources_coef):
        wn_scores, structed_oov_pairs = compute_wordnet_path_scores(pairs)

        data = np.stack((scores, wn_scores), axis=1)

        ## scale to mean==0, stddev==1
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        scaled_data = sc.fit_transform(data)

        scores = list(scaled_data[:, 0])
        wn_scores = list(scaled_data[:, 1])

        ## cleanup: replace nan with None
        scores = [i if not np.isnan(i) else None for i in scores]
        wn_scores = [i if not np.isnan(i) else None for i in wn_scores]

        new_scores = []
        for index, pair in enumerate(pairs):

            if wn_scores[index] is None:
                # path = wn_mean
                path = scores[index]  # keep the scores index for both parts!!
            else:
                path = wn_scores[index]

            new_scores.append(structed_sources_coef * scores[index] + (1 - structed_sources_coef) * path)

        return np.array(new_scores)
