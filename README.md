# Natural Language Processing of Cyberbullying Tweets 
Cyberbullying is a serious problem that can have detrimental effects on individuals' mental health and well-being. Given the large volume of tweets generated daily, manually identifying cyberbullying is time-consuming and inefficient. This tool has been developed to address this challenge by efficiently flagging potentially harmful tweets, aiming to create a safer online environment for all.

- Used Kaggle's Cyberbullying Dataset with under-sampling.
- Cleaned and transformed data for machine learning.
- Trained various models using cross-validation.
- Built a user-friendly API with Flask.


## Code and Resources
**Python Version:** 3.10  
**Packages:** numpy, pandas, nltk, scikit-learn, xgboost, flask, json, pickle  
**Original Dataset:** https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification  
**Setting Up Environment:** 
- ```conda create -p venv python=3.10 -y```  
- ```pip install -r requirements.txt```

## Getting Data
The dataset utilized, "cyberbullying_tweets_v2.csv," is crafted through under-sampling of the <a href="https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification">Cyberbullying Classification Dataset</a> from Kaggle. This dataset is tailored for binary classification, distinguishing potentially harmful tweets from non-cyberbullying ones.

![alt text](https://github.com/polaternez/cyberbullying-tweet-detection/blob/master/reports/figures/cyberbullying_type_counts.jpg "Cyberbullying Type Counts")

## Data Cleaning
A Python script is employed for comprehensive text data cleaning, encompassing operations such as:

- Punctuation removal
- Numeric character removal
- Lowercasing
- Stop word elimination
- Lemmatization/Stemming
- URL removal


## Model Building 
1. **Data Splitting:** Segregation of data into training and testing sets with a 70-30 ratio.
2. **Feature Engineering:** Utilization of a transformation pipeline integrating CountVectorizer and TfidfTransformer.
3. **Model Training and Evaluation:** Training multiple models via cross-validation, with selection based on accuracy and training efficiency. The XGBClassifier model is chosen for its superior performance.
4. **Fine-tuning:** Refinement of the XGBClassifier model for optimal performance.

After cross-validation, the models show the following performances:

![alt text](https://github.com/polaternez/cyberbullying-tweet-detection/blob/master/reports/figures/models.png "Model Performances")


## Productionization 
The project culminates in the development of a user interface facilitated by Flask, along with an API endpoint for real-time cyberbullying detection.

![alt text](https://github.com/polaternez/cyberbullying-tweet-detection/blob/master/reports/figures/flask_api.png "Cyberbullying Tweet Detector API")




