from flask import Flask,request, url_for, redirect, render_template
import joblib
import numpy as np


model=joblib.load('model2.joblib')

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('index.html',pred='Your Loan will Approve.\nProbability of Loan Approval is {}'.format(output))
    else:
        return render_template('index.html',pred='Your Loan will not Approve.\nProbability of Loan not Approval is {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
