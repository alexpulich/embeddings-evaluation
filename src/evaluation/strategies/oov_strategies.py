import logging

import deepcut
import numpy as np

from .base import OOVStrategy

logger = logging.getLogger(__name__)


class NoActionOOVStrategy(OOVStrategy):
    def handle_oov(self, embeddings, X, words):
        pass


class DeepcutOOVStrategy(OOVStrategy):
    def handle_oov(self, embeddings, X, words):
        oov_vecs_created = 0
        info_created_words = {}
        info_oov_words = {}

        # creating a set of OOV words
        oov_words = set()

        for query in X:
            for query_word in query:
                if query_word not in words:
                    oov_words.add(query_word)

        # iterating through OOV words to get AVG vectors for them
        for ds_word in oov_words:
            tokens = deepcut.tokenize(ds_word)
            in_voc_tokens = [token for token in tokens if token in embeddings]

            ## if we found word-parts in the emb - use their vectors (avg) to represent the OOV word
            if in_voc_tokens:
                token_vecs = [embeddings.get(t) for t in in_voc_tokens]
                embeddings[ds_word] = np.mean(token_vecs, axis=0)
                oov_vecs_created += 1
                info_created_words[ds_word] = in_voc_tokens
            else:
                info_oov_words[ds_word] = tokens

        logger.debug('All OOV words after deepcut:')
        logger.debug(info_oov_words)
        logger.debug('All "created"/replaced words by deepcut:')
        logger.debug(info_created_words)


class LettersCutOOVStrategy(OOVStrategy):
    def handle_oov(self, embeddings, X, words):
        oov_vecs_created = 0
        info_created_words = {}
        info_oov_words = {}

        oov_words = set()

        # collecting oov words
        for query in X:
            for query_word in query:
                if query_word not in words:
                    oov_words.add(query_word)

        # iterating through each oov-word
        for oov_word in oov_words:
            cut_word = oov_word
            words_with_same_prefix = set()

            # cutting letter by letter until we find some words with the same prefix
            while len(cut_word) and cut_word not in words:
                cut_word = cut_word[:-1]

                # collectings words with the same prefix
                for vocabulary_word in embeddings:
                    if vocabulary_word[0].startswith(cut_word):
                        words_with_same_prefix.add(vocabulary_word[0])

                # if found at least one word, then stop cutting and let's compute the avg vector
                if len(words_with_same_prefix):
                    break
            logger.debug(f'FOR WORD {oov_word} FOUND WORDS WITH THE SAME PREFIX: {str(words_with_same_prefix)}')
            if words_with_same_prefix:
                token_vecs = [embeddings.get(t) for t in words_with_same_prefix]
                embeddings[oov_word] = np.mean(token_vecs, axis=0)
                oov_vecs_created += 1
                info_created_words[oov_word] = cut_word
