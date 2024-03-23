import pickle

import numpy as np
from flask import Flask, request, render_template

app  = Flask(__name__)


#with open('model_pickle_model.pkl', 'rb') as file:
# loaded_model = pickle.load(file)

model= pickle.load(open('model_pickle_model.pkl','rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    input_text = request.form['text']
    input_text_sp = input_text.split(',')
    np_data = np.asarray(input_text_sp, dtype=np.float32)
    prediction = model.predict(np_data.reshape(1,-1))

    if prediction == 1:
        output = "This person has parkinson disease"
    else:
        output = "This person does not have parkinson disease"

    return render_template("index.html", message= output)

if __name__ == "__main__":
    app.run(debug=True)