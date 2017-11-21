import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def score(self, model, n):
        logL = model.score(self.X, self.lengths)
        # number of free parameters are determined based on
        # https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/12
        # init_p + transition_p + emission_p
        p = n * (n-1) + (n-1) + 2 * self.X.shape[1] * n
        return (-2 * logL) + (p * np.log(self.X.shape[0]))

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # init lowest score to the worst possible score
        best_score = float('inf')
        best_model = None

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                model_score = self.score(model, n)
                if model_score < best_score:
                    best_score = model_score
                    best_model = model
            except:
                pass

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def score(self, model):
        other_score = 0.0
        # iterate through all words and add together all their scores
        for word in self.words:
            if word != self.this_word:
                X, lengths = self.hwords[word]
                other_score += model.score(X, lengths)

        return model.score(self.X, self.lengths) - (other_score/(len(self.words)-1))

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # init the best score and best model
        best_score = float('-inf')
        best_model = None

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                model_score = self.score(model)
                if model_score > best_score:
                    best_score = model_score
                    best_model = model
            except:
                pass

        return best_model



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def score(self, n):
        test_scores = []
        # default n_splits is 3, we will take min of num_sequences and default
        split_method = KFold(n_splits=min(3, len(self.sequences)))

        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
            self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
            model = self.base_model(n)
            cv_test_X, cv_test_lengths = combine_sequences(cv_test_idx, self.sequences)
            test_scores.append(model.score(cv_test_X, cv_test_lengths))

        return model, np.mean(test_scores)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float("-inf")
        best_model = None

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model, model_score = self.score(n)
                if model_score > best_score:
                    best_score = model_score
                    best_model = model
            except:
                pass

        return best_model
