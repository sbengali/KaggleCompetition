#!/usr/local/bin/python

import codecs
import numpy as np
import pandas as pd
import re
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

class ExtractRecipe():
    """ 
    Class that extracts recipe information from JSON.
    """
    def __init__(self,json):
        self.recipe_id = self.set_id(json)
        self.cuisine = self.set_cuisine(json)
        self.ingredients = self.set_ingredients(json)
        self.ingredient_count = len(self.ingredients)
    def __str__(self):
        return "ID: %s\nCuisine: %s\nIngredients: %s\nNumber of Ingredients: %s" % (self.recipe_id, self.cuisine,', '.join(self.ingredients),self.ingredient_count)
    def set_id(self,json):
        """
        Method that sets the recipe id.
        """
        try:
            return json['id']
        except KeyError:
            return '-99'
    def set_cuisine(self,json):
        """
        Method that sets the recipe cuisine.
        """
        try:
            return json['cuisine']    
        except KeyError:
            return ''
    def set_ingredients(self,json):
        """
        Method that sets the recipe ingredients.
        """
        try:
            return json['ingredients']
        except KeyError:
            return []
    def clean_ingredient(self,s):
    	"""
    	Method that returns a cleaned up version of the entered ingredient.
    	"""
    	# replace unicode character with the closest ascii representation
    	from unidecode import unidecode
    	s = unidecode(s)
    	# remove parentheses and enclosed strings, e.g., "(14 oz.)"
    	s = re.sub(r'\([^)]*\)', '', s)
    	# remove punctuations, such as semicolons and commas
    	s = re.sub('[^A-Za-z0-9]+', '', s)
    	# use nltk POS tagger to find the nouns
    	import nltk
    	processed = list()
    	tokens = nltk.word_tokenize(s)
    	tagged = nltk.pos_tag(tokens)
    	for item in tagged:
    		if item[1][0] == 'N':
    			processed.append(item[0])
    	s = ''.join(processed)
    	return s
    def get_train(self):
        """
        Method that returns a dictionary of data for the training set.
        """
        return {
            'cuisine': self.cuisine,
            'ingredients': ', '.join([self.clean_ingredient(x) for x in self.ingredients]),
            'ingredient_count': self.ingredient_count
        }
    def get_predict(self):
        """
        Method that returns a dictionary of data for predicting recipes.
        """
        return {
            'id': self.recipe_id,
            'ingredients': ', '.join([self.clean_ingredient(x) for x in self.ingredients]),
            'ingredient_count': self.ingredient_count
        }
        
def loadTrainSet(dir='../input/train.json'):
	"""
	Read in JSON to create training set.
	"""
	import json
	from pandas import DataFrame, Series
	from sklearn.preprocessing import LabelEncoder
	X = DataFrame([ExtractRecipe(x).get_train() for x in json.load(codecs.open(dir, 'r', encoding='utf-8'))])
	encoder = LabelEncoder()
	X['cuisine'] = encoder.fit_transform(X['cuisine'])
	return X, encoder
	
def transformString(s):
	return ', '.join([''.join(y.lower() for y in x if y.isalnum()) for x in s.split(',')])

linear_svc_pipe = Pipeline([
	('tfidf', TfidfVectorizer(strip_accents='unicode',analyzer="char",preprocessor=transformString)),
    ('feat', SelectPercentile(chi2)),
    ('model', LogisticRegression())
])

linear_svc_grid = {
    'tfidf__ngram_range':[(3,5)],
    'feat__percentile':[90],
    'model__C':[1]
}

def fitLinearSVC(X,y,cv,i,model):
	from sklearn.metrics import accuracy_score
	tr = cv[i][0]
	vl = cv[i][1]
	model.fit(X.iloc[tr],y.iloc[tr])
	pred = model.predict(X.iloc[vl])
	score = accuracy_score(y.iloc[vl], pred)
	return  {"score": score}
	
def trainLinearSVC(model, grid, train, target, cv, refit=True, n_jobs=-1):
	from joblib import Parallel, delayed   
	from sklearn.grid_search import ParameterGrid
	from numpy import zeros
	pred = zeros((train.shape[0], target.unique().shape[0]))
	best_score = 0
	for g in ParameterGrid(grid):
		model.set_params(**g)
		results = Parallel(n_jobs=n_jobs)(delayed(fitLinearSVC)(train, target, list(cv), i, model) for i in range(cv.n_folds))
		total_score = 0
		for i in results:
			total_score += i['score']
		avg_score = total_score / len(results)
		if avg_score > best_score:
			best_score = avg_score
			best_grid = g
	print "Best Score: %0.5f" % best_score 
	print "Best Grid", best_grid
	if refit:
		model.set_params(**best_grid)
		model.fit(train, target)
	return model

def main():
	# load data
	train, encoder = loadTrainSet()
	cv = KFold(train.shape[0], n_folds=8, shuffle=True)
	
	trainLinearSVC(linear_svc_pipe, linear_svc_grid, train.ingredients, train.cuisine, cv)
	
main()
