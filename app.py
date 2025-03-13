import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import pandas as pd 
import numpy as np
import streamlit as st

## Loading the trained model
model = tf.keras.models.load_model('model.h5')

## Load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)   

## Streamlit app
st.title("Customer Churn Prediction with ANN")

## User Inputs
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Create input data as a dictionary
input_data = {
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

# Convert dictionary to DataFrame
input_data = pd.DataFrame(input_data)

## Use OneHot Encoded 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

## Concatenate encoded columns with input data dataframe
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

## Scaling the data
input_scaled = scaler.transform(input_data)

## Predict Churn
prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]

st.title(f'Churn Probability:{prediction_prob:.2f}')

if prediction_prob > 0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")
