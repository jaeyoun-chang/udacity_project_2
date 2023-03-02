import json
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
import plotly
from plotly.graph_objs import Bar, Pie

import joblib
# from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)

def tokenize(text):
    '''
    Function to tokenize function text data including lemmatizing, 
    normalizing, and removing white space
    - text: str, text messages
    '''   
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse.db', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_names = df.columns[4:]
    category_counts = (df.iloc[:, 4:] != 0).sum().values
    category_percent = category_counts / category_counts.sum()
    
    # create visuals

    graphs = [
        # Graph on Genre
        {
            'data': [
                Bar(
                    x = genre_names,
                    y = genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': 'Genre'
                }
            }
        },

        # Graph on Category - Pie
        {
            'data': [
                Pie(
                    labels = category_names,
                    values = category_percent
                )
            ],
            'layout': {
                'title': 'Percentage of Message Categories', 'height': 500
                },
                'textinfo': 'label + percent',
                'textposition': 'outside',
        },

        # # Graph on Category - Dist
        # {
        #     'data': [
        #         Bar(
        #             x = category_names,
        #             y = category_counts
        #         )
        #     ],
        #     'layout': {
        #         'title': 'Distribution of Message Categories',
        #         'yaxis': {
        #             'title': 'Count'
        #         },
        #         'xaxis': {
        #             'title': 'Messages Category',
        #             'tickangle' : 30
        #         }
        #     }
        # }

        ]
        
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()