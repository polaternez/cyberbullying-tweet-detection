import json
from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import text_cleaning as tc


app = Flask(__name__)
model = pickle.load(open("models/final_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    new_data = request.form["tweet_text"]
    temp_df = pd.DataFrame(np.array(new_data).reshape(-1, 1), columns=['tweet_text'])
    clean_text = temp_df["tweet_text"].apply(tc.clean_data)
    
    prediction = model.predict(clean_text)
    output = ""

    if prediction[0]==0:
        output = "#not_cyberbullying"
    elif prediction[0]==1:
        output = "#cyberbullying"
    else:
        output = "Error!!"

    return render_template("index.html", prediction_text=output, tweet=f"<< {new_data} >>")
        
@app.route("/predict_api", methods=["POST"])
def predict_api():
    '''
    For direct API calls trought request
    '''
    request_json = request.get_json()
    new_data = request_json["input"]
    temp_df = pd.DataFrame(np.array(new_data).reshape(-1, 1), columns=['tweet_text'])
    clean_text = temp_df["tweet_text"].apply(tc.clean_data)

    prediction = model.predict(clean_text)
    output = prediction
    
    response = json.dumps({'response': str(output)})
    return response, 200

if __name__ == "__main__":
    app.run(debug=True)


