# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 21:03:12 2024

@author: Mrinal Kalita
"""

import pickle
import numpy as np
import streamlit as st


scl = pickle.load(open('scaled_model.sav','rb'))
model = pickle.load(open('finalized_model.sav','rb'))

flatten_list = lambda y:[x for a in y for x in flatten_list(a)] if type(y) is list else [y]


def pred(input_data):
    input_data_array = flatten_list(input_data)
    input_data_scaled = scl.transform([input_data_array])
    input_data_reshape = input_data_scaled.reshape(1,-1)
    prediction = model.predict(input_data_reshape)
    print(prediction)
    
    if prediction[0] == 1:
        return "The patient is diabetic."
    else:
        return 'The patient is not diabetic.'
    
def main():
    st.title("Diabetes Prediction")
    st.header("Enter the value of the following parameters:")
    age	 = st.number_input("Enter age")
    hypertension	 = st.selectbox("hypertension",(1,0))
    heart_disease	 = st.selectbox("heart_disease",(1,0))
    bmi	 = st.number_input("Enter bmi")
    HbA1c_level = st.number_input("Enter HbA1c_level")
    blood_glucose_level = st.number_input("Enter blood_glucose_level")
    gender = st.selectbox("gender",('Female','Male','Other'))
    smoking_history = st.selectbox("smoking_history",('No_Info','never','ever','not current','current','former'))
    
    if gender == 'Female':
        gender = [1,0,0]
    elif gender == 'Male':
        gender = [0,1,0]
    else:
        gender = [0,0,1]

    if smoking_history =='No_Info':
        smoking_history = [1,0,0,0,0,0]
    elif smoking_history == 'current':
        smoking_history = [0,1,0,0,0,0]
    elif smoking_history == 'ever':
        smoking_history = [0,0,1,0,0,0]
    elif smoking_history == 'former':
        smoking_history = [0,0,0,1,0,0]
    elif smoking_history == 'never':
        smoking_history = [0,0,0,0,1,0]
    else:
        smoking_history = [0,0,0,0,0,1]
        
        
    diabetes =  ' '
    
    if st.button('Predict Patient condition'):
        diabetes=pred([age,hypertension,heart_disease,bmi,HbA1c_level,blood_glucose_level,gender,smoking_history])
    st.success(diabetes)
    
if __name__ == '__main__':
    main()