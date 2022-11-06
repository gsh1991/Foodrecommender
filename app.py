# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:59:41 2022

@author: Gopinaath
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd

import pickle

app = Flask(__name__)
model = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    MealType = int(request.form['PreferredMealType'])
    FoodType = int(request.form['PreferredFoodType'])
    MainIngredient = int(request.form['MainIngredient'])
    Highpreferred = int(request.form['HighlypreferredIngredient'])
    Lowpreferred = int(request.form['LowpreferredIngredient'])
    features = pd.DataFrame(columns=['MealType','FoodType','MainIngredient','Highpreferred','Lowpreferred'])
    features['MealType'] = MealType
    features['FoodType'] = FoodType
    features['MainIngredient'] = MainIngredient
    features['Highpreferred'] = Highpreferred
    features['Lowpreferred'] = Lowpreferred
    prediction = model.predict([[MealType,FoodType,MainIngredient,Highpreferred,Lowpreferred]])
    
    return render_template('index.html', prediction_text=prediction)
    

    
    #return render_template('result.html', prediction ='Recommendedfoods'.format(prediction))
    
                    
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
    
    