# Response-Classification

## Project

This repository is for the second project of the data scientist program of Udacity.

The given dataset has pre-labelled tweet and messages of disaster situation.  
And the project is to build a ML model that processes natural language texts  
and allocates them into 36 categories.  

## Files  

The project is composed of the following main files:  

* ETL pipeline to extract data from the given dataset and to save them as  
  a database after pre-processing:
  ETL Pipeline Preparation.ipynb  
  messages.csv   
  categories.csv   
  DisasterResponse.db
 
* ML Pipeline to build and train a model to classify texts into categories  
  ML Pipeline Preparation.ipynb
  train_classifier.py (script to run ipynb above)

* Web App to activate the model 

## Installation

* Clone this repository  
  (Please be well noted that **classifier.pkl** is NOT included in this repository)

* Install modules including:  
  ML: Pandas, Sciki-Learn  
  Natural Language Processing: NLTK  
  SQL: SQLalchemy  
  Web App and Data Visualization: Flask, Plotly  

## Web App Activation

* Run the following commands in the project's root directory to set up your database and model

    - To run ETL pipeline that cleans data and stores in database  
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves  
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

* Run the following command in the app's directory to run your web app.
    `python run.py`

* Go to http://0.0.0.0:3001/

## Acknowledgements

* Udacity for providing a great Data Science course  
* Figure Eight(https://www.figure-eight.com/) for datasets

## License

* This repository is with MIT License.