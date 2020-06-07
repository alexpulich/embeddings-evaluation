from evaluation.strategies.oov_strategies import DeepcutOOVStrategy, LettersCutOOVStrategy
from evaluation.strategies.wordnet.strategies import WordNetMethod1Strategy, WordNetMethod2Strategy
from evaluation.strategies.conceptnet.strategies import ConceptNetMethod1Strategy, ConceptNetMethod2Strategy

CLI_OOV_OPTION = {
    'deepcut': DeepcutOOVStrategy,
    'letters': LettersCutOOVStrategy
}

CLI_SS_OPTION = {
    'wn1': WordNetMethod1Strategy,
    'wn2': WordNetMethod2Strategy,
    'cn1': ConceptNetMethod1Strategy,
    'cn2': ConceptNetMethod2Strategy
}

