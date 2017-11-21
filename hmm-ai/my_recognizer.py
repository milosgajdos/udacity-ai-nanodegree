import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key is a word and
       value is Log Likelihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for test_word, (test_X, test_lengths) in test_set.get_all_Xlengths().items():
        best_score = float("-inf")
        best_guess = ""
        p = {}

        for word, model in models.items():
            try:
                p[word] = model.score(test_X, test_lengths)
            except:
                p[word] = float("-inf")

            if p[word] > best_score:
                best_score = p[word]
                best_guess = word

        probabilities.append(p)
        guesses.append(best_guess)

    return probabilities, guesses
