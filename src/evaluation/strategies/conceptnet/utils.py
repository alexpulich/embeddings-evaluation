from config import CONCEPTNET_PICKLE_FILE


def get_similarity_from_dict(scores, word1, word2):
    return scores.get(word1 + word2)


def compute_conceptnet_path_scores(pairs, numberbatch=None):
    """
    Compute ConceptNet path similarity for a list of input word pairs
    If we don't find a path between the two works, we add "None" to the result list
    @returns: this list of simility scores, and the number of OOV-word-pairs
    """

    oov_pairs = 0  # we count word pairs for which we have no path
    scores = []
    if not numberbatch:
        import pickle
        with open(CONCEPTNET_PICKLE_FILE, 'rb') as f:
            model = pickle.load(f)
        for index, pair in enumerate(pairs):
            score = get_similarity_from_dict(model, pair[0], pair[1])
            scores.append(score)
            if score is None:
                oov_pairs += 1
    else:
        for index, pair in enumerate(pairs):
            score = numberbatch.get_similarity(pair[0], pair[1])
            scores.append(score)
            if score is None:
                oov_pairs += 1

    return scores, oov_pairs
