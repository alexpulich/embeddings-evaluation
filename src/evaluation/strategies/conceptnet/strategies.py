import logging

import numpy as np

from ..base import StructuredSourceStrategy
from .numerbatch import ConceptNetNumberbatch
from .utils import compute_conceptnet_path_scores


logger = logging.getLogger(__name__)


class ConceptNetMethod1Strategy(StructuredSourceStrategy):
    """
    Method for the combination of WE and CN scores
    Basic idea: have a coefficient to weight the influence of the two components
    If we don't have a path similarity for a word pair, we use the average path_similarity.
    """

    def apply(self, scores, pairs, structed_sources_coef):
        cn_scores, structed_oov_pairs = compute_conceptnet_path_scores(pairs, ConceptNetNumberbatch)
        cn_mean = np.mean(np.array([cn_score for cn_score in cn_scores if cn_score is not None]))
        logger.debug(f"conceptnet method 1: avg path similarity: {cn_mean}")

        new_scores = []
        for index, pair in enumerate(pairs):
            if cn_scores[index] is None:
                path = cn_mean
            else:
                path = cn_scores[index]

            new_scores.append(structed_sources_coef * scores[index] + (1 - structed_sources_coef) * path)

        return np.array(new_scores)


class ConceptNetMethod2Strategy(StructuredSourceStrategy):
    """
    Method 2 for the combination of WE and CN scores
    Basic idea: have a coefficient to weight the influence of the two components
    If we don't have a path similarity for a word pair, we use only WE similarity
    Here we transform both the list WE-scores and WC-scores to have mean==0, stddev==1
        -- in order to have them on the same scale when combining them
    """

    def apply(self, scores, pairs, structed_sources_coef):
        cn_scores, structed_oov_pairs = compute_conceptnet_path_scores(pairs, ConceptNetNumberbatch)

        data = np.stack((scores, cn_scores), axis=1)

        ## scale to mean==0, stddev==1
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        scaled_data = sc.fit_transform(data)

        scores = list(scaled_data[:, 0])
        cn_scores = list(scaled_data[:, 1])

        ## cleanup: replace nan with None
        scores = [i if not np.isnan(i) else None for i in scores]
        cn_scores = [i if not np.isnan(i) else None for i in cn_scores]

        new_scores = []
        for index, pair in enumerate(pairs):

            if cn_scores[index] is None:
                # path = wn_mean
                path = scores[index]  # keep the scores index for both parts!!
            else:
                path = cn_scores[index]

            new_scores.append(structed_sources_coef * scores[index] + (1 - structed_sources_coef) * path)

        return np.array(new_scores)
