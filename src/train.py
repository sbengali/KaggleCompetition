#!/usr/local/bin/python

import codecs
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectPercentile, f_classif
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

class ExtractRecipe():
    """ 
    Class for extracting data from JSON input file
    """

    def __init__(self, json):
        self.recipe_id = self.set_id(json)
        self.cuisine = self.set_cuisine(json)
        self.ingredients = self.set_ingredients(json)
        self.ingredient_count = len(self.ingredients)

    def __str__(self):
        return "ID: %s\nCuisine: %s\nIngredients: %s\nNumber of Ingredients: %s" % (
        self.recipe_id, self.cuisine, ', '.join(self.ingredients), self.ingredient_count)

    def set_id(self, json):
        """
        Method that sets the recipe id.
        """
        try:
            return json['id']
        except KeyError:
            return '-99'

    def set_cuisine(self, json):
        """
        Method that sets the recipe cuisine.
        """
        try:
            return json['cuisine']
        except KeyError:
            return ''

    def set_ingredients(self, json):
        """
        Method that sets the recipe ingredients.
        """
        try:
            return json['ingredients']
        except KeyError:
            return []

    def get_train(self):
        """
        Create a dictionary of data from training set
        """
        return {
            'cuisine': self.cuisine,
            'ingredients': ', '.join([x for x in self.ingredients]),
            'ingredient_count': self.ingredient_count
        }

    def get_predict(self):
        """
        Method that returns a dictionary of data for predicting recipes.
        """
        return {
            'id': self.recipe_id,
            'ingredients': ', '.join([x for x in self.ingredients]),
            'ingredient_count': self.ingredient_count
        }

def loadTrainSet(dir='../input/nltk_filtered_train_forcetags.json'):
    """
    Read training data from JSON
    """
    import json
    from pandas import DataFrame, Series
    from sklearn.preprocessing import LabelEncoder
    X = DataFrame([ExtractRecipe(x).get_train() for x in json.load(open(dir, 'rb'))])
    encoder = LabelEncoder()
    X['cuisine'] = encoder.fit_transform(X['cuisine'])
    return X, encoder

def loadTestSet(dir='../input/nltk_filtered_test_forcetags.json'):
    """
    Read test data from JSON
    """
    import json
    from pandas import DataFrame
    X = DataFrame([ExtractRecipe(x).get_predict() for x in json.load(open(dir, 'rb'))])
    return X

classification_pipe = Pipeline([
    #TF-IDF: analyzer: Whether the feature should be made of word or character n-grams.
    ('tfidf', TfidfVectorizer(analyzer="char", strip_accents="unicode", ngram_range=[2, 8])),
    ('feat', SelectPercentile(f_classif)),
    ('model', LogisticRegression(multi_class="multinomial", solver="lbfgs"))
])

#classification_params = {
#    'tfidf__ngram_range': [(3, 5)],
#    'feat__percentile': [95, 90, 85],
#    'model__C': [1]
#}

def fitModel(X, y, cv, i, model):
    from sklearn.metrics import accuracy_score
    tr = cv[i][0]
    vl = cv[i][1]
    model.fit(X.iloc[tr], y.iloc[tr])
    pred = model.predict(X.iloc[vl])
    score = accuracy_score(y.iloc[vl], pred)
    return {"score": score}

# Called by main method in run.py
def trainModel(model, train, target, cv, refit=True, n_jobs=-1):
    from joblib import Parallel, delayed
    from sklearn.grid_search import ParameterGrid
    from numpy import zeros
    pred = zeros((train.shape[0], target.unique().shape[0]))
    best_score = 0
    #for g in ParameterGrid(grid):
        #model.set_params(**g)
    results = Parallel(n_jobs=n_jobs)(delayed(fitModel)(train, target, list(cv), i, model) for i in range(cv.n_folds))
    print "score results"
    print results
    total_score = 0
    for i in results:
        total_score += i['score']
    avg_score = total_score / len(results)
    if avg_score > best_score:
        best_score = avg_score
       #best_grid = g
    print "Best Score: %0.5f" % best_score
    #print "Best Grid", best_grid
    if refit:
        #model.set_params(**best_grid)
        model.fit(train, target)
    return model



	


