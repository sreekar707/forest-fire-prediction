import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from flask import Flask,jsonify,render_template,request

app = Flask(__name__)

# Load models using absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
ridge_model = pickle.load(open(os.path.join(current_dir, 'models', 'ridge.pkl'), 'rb'))
standard_scaler = pickle.load(open(os.path.join(current_dir, 'models', 'scaler.pkl'), 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        temperature = float(request.form.get('temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        
        new_data_scaled=standard_scaler.transform([[temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)
        
        return render_template('home.html',results=result[0])
    
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(debug=True)