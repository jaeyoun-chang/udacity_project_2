# import libraries

import sys
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

def load_data(database_filepath):
    '''
    Function to read SQLite database and split dataframe into input & output
    - database_filepath: str, the file path of SQLite database
    '''

    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table(database_filepath, engine)
    # replace the value 2 of columns related to 0
    # as they are not disaster messages
    # (ASK A MENTOR: https://knowledge.udacity.com/questions/954962)
    df.related = df.related.replace(2, 0)
    X = df.message.values
    Y = df.iloc[:, 4:].values
    target_name = df.columns[4:]
    
    return X, Y, target_name

def tokenize(text):
    '''
    Function to tokenize function text data including lemmatizing, 
    normalizing, filtering stop words, and removing white space
    - text: str, text messages
    '''
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # add stop_words
    stop_words = stopwords.words("english") + list(string.punctuation)
    
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        if tok not in stop_words:       
            # lemmatize, normalize case, and remove leading/trailing white space
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)
    
    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    Class of extracting the starting verb of a sentence,
    to create a new feature to be added in modelling
    '''

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    '''
    Function to build a ML pipeline with feature union.returns a ML Pipeline that process text messages
    according to NLP best-practice and apply a classifier.
    '''
    pipeline = Pipeline([
    ('features', FeatureUnion(
        [('text_pipeline',
          Pipeline([
              ('vect', CountVectorizer(tokenizer=tokenize)),
              ('tfidf', TfidfTransformer()),
              ])),
         ('isverb', StartingVerbExtractor())
         ])),
    ('clf', MultiOutputClassifier(
        RandomForestClassifier(
        class_weight='balanced',
        n_jobs = -1
        )))
    ])


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()