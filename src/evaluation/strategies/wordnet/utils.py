from pythainlp.corpus import wordnet
from config import WORDNET_PATH_SIMILARITY_TYPE

def compute_wordnet_path_scores(pairs):
    """
        Compute WordNet path similarity for a list of input word pairs
        Note: Thai WordNet has 3 methods to compute a similarity value: wordnet.path_similarity, wordnet.lch_similarity, wordnet.wup_similarity
            lch_similarity we can't use. path_similarity seems to have better results than wup_similarity

        If we don't find a path between the two works, we add "None" to the result list

        @returns: this list of simility scores, and the number of OOV-word-pairs
    """

    structed_oov_pairs = 0
    wn_scores = []

    for index, pair in enumerate(pairs):

        w1 = wordnet.synsets(pair[0])
        w2 = wordnet.synsets(pair[1])

        if len(w1) > 0 and len(w2) > 0:
            # just use the first synset of each term
            if WORDNET_PATH_SIMILARITY_TYPE == 'first_synset':
                path = wordnet.path_similarity(w1[0], w2[0])

            # return the highest sim between all synset combinations
            elif WORDNET_PATH_SIMILARITY_TYPE == 'most_similar':
                path = -1
                for syn1 in w1:
                    for syn2 in w2:
                        tmppath = wordnet.path_similarity(syn1, syn2)
                        if tmppath and tmppath > path: path = tmppath
                if path == -1:
                    # if no path found, set back to None
                    path = None
            else:
                raise RuntimeError('WORDNET_PATH_SIMILARITY_TYPE is not set in config!')

            wn_scores.append(path)
        else:
            wn_scores.append(None)
            structed_oov_pairs += 1

    return wn_scores, structed_oov_pairs
