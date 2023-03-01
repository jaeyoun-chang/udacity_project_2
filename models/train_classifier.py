# import libraries
import sys
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
from sklearn.model_selection import train_test_split, GridSearchCV
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

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(database_filepath, engine)
    # replace the value 2 of columns related to 0
    # as they are not disaster messages
    # (ASK A MENTOR: https://knowledge.udacity.com/questions/954962)
    df.related = df.related.replace(2, 0)
    X = df.message.values
    Y = df.iloc[:, 4:].values
    category_names = df.columns[4:]       
    return X, Y, category_names

def tokenize(text):
    '''
    Function to tokenize function text data including lemmatizing, 
    normalizing, and removing white space
    - text: str, text messages
    '''    
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
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

def build_model(X_train, Y_train):
    '''
    Function to build ML pipeline with feature union and
    return Gird_Search_CV with the best parameters
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
    pipeline.fit(X_train, Y_train)
    # Hyper-parameters of grid_search
    parameters = {
    'clf__estimator__n_estimators': [100, 300, 500],
    'clf__estimator__min_samples_split': [2, 6, 10, 14, 18],
    }
    # Create a grid search object using the pipeline and parameters
    model = GridSearchCV(pipeline, parameters, cv = 3, verbose = 3)  
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to report the f1 score, precision, recall and accuracy
    for each output category of the dataset by iterating through the columns
    and calling sklearn's `classification_report`
    
    - model: ML model
    - X_test: test samples of input
    - Y_test: test samples of output labels
    category_names: label names of multi-output
    '''
    
    Y_pred = model.predict(X_test)
    list_precision, list_recall, list_f1 = [], [], []

    # Iterate 36 target columns and generate a classification report for each
    for i, col in enumerate(category_names):
        
        rslt = classification_report(Y_test[:, i], Y_pred[:, i])
        
        # weighted avg scores are in the 2nd last line: 
        score_line = rslt.split('\n')[-2]
        score_line_split = score_line.split()
        
        # scores are in the 2nd to 4th places of splitted texts of score_line
        precision_score = float (score_line_split[2])
        list_precision.append(precision_score)
        
        recall_score = float (score_line_split[3])
        list_recall.append(recall_score)
        
        f1_score = float (score_line_split[4])
        list_f1.append(f1_score)
        
        print(f'{i} Target column {col}:')
        print(rslt)
        print()
        
    print ('mean of weighted avg precision: {:.2f}'.format(sum(list_precision)/len(list_precision)))
    print ('mean of weighted avg recall: {:.2f}'.format(sum(list_recall)/len(list_recall)))
    print ('mean of weighted avg f1: {:.2f}'.format(sum(list_f1)/len(list_f1)))
    # accuracy is the same as weighted ave recall
    print ('mean of accuracy: {:.2f}'.format(sum(list_recall)/len(list_recall)))

# def save_model(model, model_filepath):
def save_model(model, model_filepath):
    '''
    Function to save trained model as Pickle file
    - model: ML model
    - model_filepath: str, the file path of Pickle file
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train, Y_train)
        
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