# Cyberbullying Tweet Detector: Project Overview  
* This tool was created to flag potentially harmful tweets
* Take Cyberbullying Classification Dataset from Kaggle then build our train data with under-sampling technique.
* Cleaning the data
* Build transformation pipeline
* Train different models and evaluate them using cross validation.
* Built a client facing API using Flask 

Note: This project was made for educational purposes.

## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** numpy, pandas, nltk, sklearn, xgboost, flask, json, pickle  
**For Flask API Requirements:**  ```pip install -r requirements.txt```  
**For the Original Version of the Dataset:** https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

## Getting Data
We use the "cyberbullying_tweets_v2.csv" dataset created by under-sampling the <a href="https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification">Cyberbullying Classification</a> dataset from Kaggle. Our dataset created for binary classification model to flag potentially harmful tweets(cyberbullying/not_cyberbullying).

![alt text](https://github.com/polaternez/cyberbullying_tweets_proj/blob/master/images/cyberbullying_type_counts.jpg "Cyberbullying Type Counts")

## Data Cleaning
We create a python script to clear text data, its apply the following operations to the text:
* Removing Puncuatations
* Removing Numbers
* Lowecasing the data
* Remove stop words
* Lemmatize/ Stem words
* Remove URLs


## Model Building 

First, we split the data into train and test sets with a test size of 30%. After that, we create transformation pipeline with CountVectorizer and TfidfTransformer for feature extraction from text.

We train six different models and evaluate them using cross validation scores. Then we get the following results:

![alt text](https://github.com/polaternez/cyberbullying_tweets_proj/blob/master/images/models.png "Model Performances")

Finally, we choose the XGBClassifier model from xgboost, it has the highest accuracy and relatively low training time, then we fine-tune the model for better performance.

## Productionization 
In this step, I created the UI with the Flask. API endpoint help receives a request tweets and returns the results of the cyberbullying detection.

![alt text](https://github.com/polaternez/cyberbullying_tweets_proj/blob/master/images/flask_api.png "Cyberbullying Tweet Detector API")




