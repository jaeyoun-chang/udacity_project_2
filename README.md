# Disaster Response Classification

## Project

This repository (https://github.com/jaeyoun-chang/udacity_project_2)  
is for the second project of the data scientist program of Udacity.

The given dataset has pre-labelled messages of disaster situation.  
And the project is to build a ML model that processes natural language texts  
and allocates them into 36 categories.  

## Files  

The project is composed of the following main files:  

* ETL pipeline to extract data from the given dataset and to save them as  
  a database after pre-processing:
  ETL Pipeline Preparation.ipynb  
  process_data.py (script to run ipynb above)  
  messages.csv   
  categories.csv   
  DisasterResponse.db
 
* ML Pipeline to build and train a model to classify texts into categories  
  ML Pipeline Preparation.ipynb
  train_classifier.py (script to run ipynb above)

* Web App to activate the model 

## Installation

* Clone this repository  
  (Please be well noted that **classifier.pkl** is NOT included in this repository.
  It will be created in 2) running ML pipeline of Web App Activation below)

* Install modules including:  
  ML: Pandas, Sciki-Learn  
  Natural Language Processing: NLTK  
  SQL: SQLalchemy  
  Web App and Data Visualization: Flask, Plotly  

## Web App Activation

* Run the following commands in root directory to set up database and model

    1) To run ETL pipeline that cleans data and stores in database  
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    2) To run ML pipeline that trains classifier and saves  
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`  
        **(This can take more than 1 hour!)**

* Go to `app` directory by `cd app` to run run.py by `python run.py`  

* Open the URL 'http://0.0.0.0:3001/' in any browser

## Acknowledgements

* Udacity for providing a great Data Science course  
* Figure Eight(https://www.figure-eight.com/) for datasets

## License

* This repository is with MIT License.