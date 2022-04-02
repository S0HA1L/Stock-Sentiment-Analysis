import numpy as np
from flask import Flask,render_template,request,jsonify
import pickle


app = Flask(__name__)
model = pickle.load(open('Sentiment_Analysis.pkl','rb'))
cv = pickle.load(open('transform3.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # for rendering results on HTML APi
    if request.method == "POST":
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        myprediction = model.predict(vect)
    return render_template('result.html', prediction = myprediction)


if __name__ == '__main__':
    app.run(debug=True)
























