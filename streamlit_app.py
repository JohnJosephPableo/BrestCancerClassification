import keras
import streamlit as st
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
model = load_model('./breastcancer_classification_model.h5')

st.title("Breast Cancer Classification")

st.write("""
    A breast cancer classification using the data from fine needle aspiration (FNA) biopsy
    """)

st.write("""
            Mean
         """)
radius_m = st.number_input('Enter the mean radius', value=0.0)
texture_m = st.number_input('Enter the mean texture', value=0.0)
perimeter_m = st.number_input('Enter the mean perimeter', value=0.0)
area_m = st.number_input('Enter the mean area', value=0.0)
smoothness_m = st.number_input('Enter the mean smoothness', value=0.0)
compactness_m = st.number_input('Enter the mean compactness', value=0.0)
concavity_m = st.number_input('Enter the mean concavity', value=0.0)
concave_points_m = st.number_input('Enter the mean concave points', value=0.0)
symmetry_m = st.number_input('Enter the mean symmetry', value=0.0)
fractal_dimension_m = st.number_input('Enter the mean fractal dimension', value=0.0)
st.write("""
            Standard Error
         """)
radius_se = st.number_input('Enter the standard error radius', value=0.0)
texture_se = st.number_input('Enter the standard error texture', value=0.0)
perimeter_se = st.number_input('Enter the standard error perimeter', value=0.0)
area_se = st.number_input('Enter the standard error area', value=0.0)
smoothness_se = st.number_input('Enter the standard error smoothness', value=0.0)
compactness_se = st.number_input('Enter the standard error compactness', value=0.0)
concavity_se = st.number_input('Enter the standard error concavity', value=0.0)
concave_points_se = st.number_input('Enter a concave points', value=0.0)
symmetry_se = st.number_input('Enter the standard error symmetry', value=0.0)
fractal_dimension_se = st.number_input('Enter the standard error fractal dimension', value=0.0)
st.write("""
            Worst
         """)
radius_w = st.number_input('Enter the worst mean', value=0.0)
texture_w = st.number_input('Enter the worst texture', value=0.0)
perimeter_w = st.number_input('Enter the worst perimeter', value=0.0)
area_w = st.number_input('Enter the worst area', value=0.0)
smoothness_w = st.number_input('Enter the worst smoothness', value=0.0) 
compactness_w = st.number_input('Enter the worst compactness', value=0.0)
concavity_w = st.number_input('Enter the worst concavity', value=0.0)
concave_points_w = st.number_input('Enter the worst concave points', value=0.0)
symmetry_w = st.number_input('Enter the worst symmetry', value=0.0)
fractal_dimension_w = st.number_input('Enter the worst fractal dimension', value=0.0)

input = np.array([radius_m,texture_m,perimeter_m,area_m,smoothness_m,compactness_m,concavity_m,concave_points_m,symmetry_m,fractal_dimension_m,
                 radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,
                 radius_w,texture_w,perimeter_w,area_w,smoothness_w,compactness_w,concavity_w,concave_points_w,symmetry_w,fractal_dimension_w]) 

if st.button('Classify'):
    input_data_reshaped = input.reshape(1,-1)
    input_data_std = scaler.transform(input_data_reshaped)
    prediction = model.predict(input_data_std)
    st.write(prediction)
    prediction_label = [np.argmax(prediction)]
    if(prediction_label == 0):
      st.write('The tumor is Malignant')
    else:
      st.write('The tumor is Benign')





