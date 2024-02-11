# Natural Language Processing of Cyberbullying Tweets 
Cyberbullying is a serious problem that can have detrimental effects on individuals' mental health and well-being. Given the large volume of tweets generated daily, manually identifying cyberbullying is time-consuming and inefficient. This tool has been developed to address this challenge by efficiently flagging potentially harmful tweets, aiming to create a safer online environment for all.
* Used Kaggle's Cyberbullying Dataset with under-sampling.
* Cleaned and transformed data for machine learning.
* Trained various models using cross-validation.
* Built a user-friendly API with Flask.

Note: This project was made for educational purposes.

## Code and Resources Used 
**Python Version:** 3.10  
**Packages:** numpy, pandas, nltk, scikit-learn, xgboost, flask, json, pickle  
**Original Version of the Dataset:** https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification  
**Create Anaconda environment:** 
- ```conda create -p venv python==3.10 -y```  
- ```pip install -r requirements.txt```

## Getting Data
We use the "cyberbullying_tweets_v2.csv" dataset created by under-sampling the <a href="https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification">Cyberbullying Classification</a> dataset from Kaggle. Our dataset created for binary classification model to flag potentially harmful tweets(cyberbullying/not_cyberbullying).

![alt text](https://github.com/polaternez/cyberbullying-tweet-detection/blob/master/reports/figures/cyberbullying_type_counts.jpg "Cyberbullying Type Counts")

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

![alt text](https://github.com/polaternez/cyberbullying-tweet-detection/blob/master/reports/figures/models.png "Model Performances")

Finally, we choose the XGBClassifier model from xgboost, it has the highest accuracy and relatively low training time, then we fine-tune the model for better performance.

## Productionization 
In this step, I created the UI with the Flask. API endpoint help receives a request tweets and returns the results of the cyberbullying detection.

![alt text](https://github.com/polaternez/cyberbullying-tweet-detection/blob/master/reports/figures/flask_api.png "Cyberbullying Tweet Detector API")




