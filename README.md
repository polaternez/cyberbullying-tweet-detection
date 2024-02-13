# Natural Language Processing of Cyberbullying Tweets 
Cyberbullying is a serious problem that can have detrimental effects on individuals' mental health and well-being. Given the large volume of tweets generated daily, manually identifying cyberbullying is time-consuming and inefficient. This tool has been developed to address this challenge by efficiently flagging potentially harmful tweets, aiming to create a safer online environment for all.

- Used Kaggle's Cyberbullying Dataset with under-sampling.
- Cleaned and transformed data for machine learning.
- Trained various models using cross-validation.
- Built a user-friendly API with Flask.


## Code and Resources Used 
**Python Version:** 3.10  
**Packages:** numpy, pandas, nltk, scikit-learn, xgboost, flask, json, pickle  
**Original Version of the Dataset:** https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification  
**Create Anaconda environment:** 
- ```conda create -p venv python=3.10 -y```  
- ```pip install -r requirements.txt```

## Getting Data
We use the "cyberbullying_tweets_v2.csv" dataset created by under-sampling the <a href="https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification">Cyberbullying Classification</a> dataset from Kaggle. Our dataset created for binary classification model to flag potentially harmful tweets(cyberbullying/not_cyberbullying).

![alt text](https://github.com/polaternez/cyberbullying-tweet-detection/blob/master/reports/figures/cyberbullying_type_counts.jpg "Cyberbullying Type Counts")

## Data Cleaning
We use a Python script to clean text data, applying the following operations:

* Removing Punctuations
* Removing Numbers
* Lowercasing the data
* Removing stop words
* Lemmatizing/Stemming words
* Removing URLs
  

## Model Building 
1. **Split data:** We split the data into train and test sets with a 30% test size.
2. **Feature extraction:** We create a transformation pipeline with CountVectorizer and TfidfTransformer.
3. **Model training and evaluation:** We train six models using cross-validation and choose the XGBClassifier model based on accuracy and training time.
4. **Fine-tuning:** We fine-tune the XGBClassifier model for better performance.

After cross-validation, we obtain the following model performances:

![alt text](https://github.com/polaternez/cyberbullying-tweet-detection/blob/master/reports/figures/models.png "Model Performances")


## Productionization 
We create a user interface with Flask and an API endpoint that receives tweet requests and returns cyberbullying detection results.

![alt text](https://github.com/polaternez/cyberbullying-tweet-detection/blob/master/reports/figures/flask_api.png "Cyberbullying Tweet Detector API")




