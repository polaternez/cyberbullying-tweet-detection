from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import json

from cbDetection.utils.text_cleaning import clean_text
from cbDetection.pipeline.predict import CustomData, PredictPipeline


app = Flask(__name__)

# Route for a home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Unbox data from request form
        data = CustomData(
            tweet_text=request.form.get('tweet_text')
        )
        data_df = data.get_data_as_dataframe()
        
        # data cleaning
        cleaned_text = data_df["tweet_text"].apply(clean_text)

        # predictions
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(cleaned_text)

        # output text
        output = ""
        if prediction[0] == 0:
            output = "#not_cyberbullying"
        elif prediction[0] == 1:
            output = "#cyberbullying"
        else:
            output = "Error!!"
        return render_template("home.html", prediction_text=output, tweet=f"<< {data.tweet_text} >>")
    
@app.route("/predict_api", methods=["POST"])
def predict_api():
    '''
    For direct API calls throught request
    '''
    request_json = request.get_json()
    temp_df = pd.DataFrame(request_json)

    # data cleaning
    cleaned_text = temp_df["input"].apply(clean_text)

    # predictions
    predict_pipeline = PredictPipeline()
    predictions = predict_pipeline.predict(cleaned_text)

    output = ["#cyberbullying" if x==1 else "#not_cyberbullying" for x in predictions]
    response = json.dumps({'response': output})
    return response, 200
    

if __name__ == "__main__":
    app.run(debug=True)  
    # app.run(host="0.0.0.0", port=8080)        
