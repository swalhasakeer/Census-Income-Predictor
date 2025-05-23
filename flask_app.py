from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
with open('classification_model.pkl','rb') as file:
  model = pickle.load(file)
  
with open('Scaler.pkl','rb') as file:
    scaler = pickle.load(file)
    
edu_map = {
    'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5, '11th': 6,
    '12th': 7, 'HS-grad': 8, 'Some-college': 9, 'Assoc-voc': 10, 'Assoc-acdm': 11,
    'Bachelors': 12, 'Masters': 13, 'Prof-school': 14, 'Doctorate': 15
}
sex_map = {'Male':0,'Female':1}
feature_to_log1p = ['sex']
num_cols_to_standardize = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']


@app.route('/')
def home():
  return render_template('index.html',prediction ="")

@app.route('/Predict', methods = ['POST'])
def index():
  
  input_data = {
    'age':int(request.form['age']),
    'education':request.form['education'],
    'sex':request.form['sex'],
    'capital-gain':int(request.form['capital-gain']),
    'capital-loss':int(request.form['capital-loss']),
    'hours-per-week':int(request.form['hours-per-week']),
    'marital_Married-civ-spouse': 1 if request.form['marital_Married-civ-spouse'] == 'Yes' else 0,
    'occupation_Exec-managerial': 1 if request.form['occupation_Exec-managerial'] == 'Yes' else 0
  }
  input_df = pd.DataFrame([input_data])
  
  input_df['education'] = input_df['education'].map(edu_map)
  input_df['sex'] = input_df['sex'].map(sex_map)
  # log1p transform
  input_df[feature_to_log1p] = input_df[feature_to_log1p].apply(np.log1p)
  # Standardize
  input_df[num_cols_to_standardize] = scaler.transform(input_df[num_cols_to_standardize])
  predicted_class = model.predict(input_df)
  if predicted_class == 0:
    predict = "This person's income is below 50k"
  else:
    predict = "This person's income is above 50k"
  return render_template('index.html', prediction=predict)


if __name__ == '__main__':
    app.run(debug=True)