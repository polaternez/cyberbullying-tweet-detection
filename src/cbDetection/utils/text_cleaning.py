import re
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# for cleaning tweets
lemmatizer = WordNetLemmatizer()
def clean_text(tweet: str) -> str:
    tweet = re.sub("[@&]\w*", "", tweet)
    tweet = re.sub("https?:\S*", "", tweet)
    tweet = re.sub("[^A-Za-z#]", " ", tweet)
    tweet = tweet.lower()
    tweet = [
        lemmatizer.lemmatize(word) for word in tweet.split() 
            if word not in stopwords.words("english")
    ]
    tweet = " ".join(tweet)
    return tweet