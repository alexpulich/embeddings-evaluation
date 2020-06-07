import numpy as np

from embeddings.embeddings import load_embedding

from config import NUMBERBATCH_PATH


def load_if_needed(func):
    def wrapper(cls, *args, **kwargs):
        if not cls.w:
            cls.w = load_embedding(NUMBERBATCH_PATH,
                                   format='word2vec',
                                   normalize=True,
                                   lower=False,
                                   clean_words=False,
                                   load_kwargs={})
            return func(cls, *args, **kwargs)

    return wrapper


class ConceptNetNumberbatch:
    w = None

    @classmethod
    @load_if_needed
    def get_similarity(cls, word1, word2):
        v1 = cls._get_vector(word1)
        if v1 is None:
            return None
        cls.w[word1] = v1

        v2 = cls._get_vector(word2)
        if v2 is None:
            return None
        cls.w[word2] = v2

        return v1.dot(v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    @classmethod
    @load_if_needed
    def _get_vector(cls, word):
        vector = cls.w.get(word)
        if vector is None:
            vector = cls._get_vector_for_ovv(word)
        if vector is None:
            return None
        return vector

    @classmethod
    @load_if_needed
    def _get_vector_for_ovv(cls, word):
        words = cls.w.vocabulary.word_id
        cut_word = word

        words_with_same_prefix = set()
        while len(cut_word) and cut_word not in words:
            cut_word = cut_word[:-1]
            # collectings words with the same prefix
            for vocabulary_word in cls.w:
                if vocabulary_word[0].startswith(cut_word):
                    words_with_same_prefix.add(vocabulary_word[0])
            if len(words_with_same_prefix):
                break
        if words_with_same_prefix:
            token_vecs = [cls.w.get(t) for t in words_with_same_prefix]
            return np.mean(token_vecs, axis=0)
        return None
