import streamlit as st
import pickle
import pandas as pd 
import numpy as np
with open('classification_model.pkl','rb') as file:
  model = pickle.load(file)
  
with open('Scaler.pkl','rb') as file:
    scaler = pickle.load(file)
    
st.title("Income Prediction App")

age = st.number_input("Age")
education = st.selectbox("Education",['Preschool','1st-4th','5th-6th','7th-8th','9th','10th','11th',
    '12th','HS-grad','Some-college','Assoc-voc','Assoc-acdm',
    'Bachelors','Masters','Prof-school','Doctorate'])
sex = st.selectbox("Sex",['Male','Female'])
capital_gain = st.number_input("Capital-gain")
capital_loss = st.number_input("Capital-loss")
hours_per_week = st.number_input("Hours-per-week")
marital_Married_civ_spouse = st.radio("Are you Married (civil spouse)?", ["Yes", "No"])
occupation_Exec_managerial = st.radio("Occupation is Exec-managerial?", ["Yes", "No"])

# Encoding binary categorical features
marital_Married_civ_spouse = 1 if marital_Married_civ_spouse == "Yes" else 0
occupation_Exec_managerial = 1 if occupation_Exec_managerial == "Yes" else 0

# Combine all features in the expected order
# Adjust this based on your model’s expected input format
input_df = pd.DataFrame([{
    'age': age,
    'education':education,
    'sex':sex,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'marital_Married-civ-spouse': marital_Married_civ_spouse,
    'occupation_Exec-managerial': occupation_Exec_managerial,
}])


edu_map = {
    'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5, '11th': 6,
    '12th': 7, 'HS-grad': 8, 'Some-college': 9, 'Assoc-voc': 10, 'Assoc-acdm': 11,
    'Bachelors': 12, 'Masters': 13, 'Prof-school': 14, 'Doctorate': 15
}
sex_map = {'Male':0,'Female':1}
feature_to_log1p = ['sex']
num_cols_to_standardize = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']

input_df['education'] = input_df['education'].map(edu_map)
input_df['sex'] = input_df['sex'].map(sex_map)
# log1p transform
input_df[feature_to_log1p] = input_df[feature_to_log1p].apply(np.log1p)
# Standardize
input_df[num_cols_to_standardize] = scaler.transform(input_df[num_cols_to_standardize])


if st.button("Predict Income"):
    prediction = model.predict(input_df)
    result = " Greater than 50K" if prediction[0] == 1 else " Less than 50K"
    st.success(f"Predicted Income: {result}")