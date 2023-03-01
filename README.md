# Response-Classification

## Project

This repository is for the second project of the data scientist program of Udacity.

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
  (Please be well noted that **classifier.pkl** is NOT included in this repository)

* Install modules including:  
  ML: Pandas, Sciki-Learn  
  Natural Language Processing: NLTK  
  SQL: SQLalchemy  
  Web App and Data Visualization: Flask, Plotly  

## Web App Activation

* Run the following commands in root directory to set up database and model

    - To run ETL pipeline that cleans data and stores in database  
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves  
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

* Go to `app` directory: `cd app`

* Run your web app: `python run.py`

* Click the `PREVIEW` button to open the homepage

## Acknowledgements

* Udacity for providing a great Data Science course  
* Figure Eight(https://www.figure-eight.com/) for datasets

## License

* This repository is with MIT License.