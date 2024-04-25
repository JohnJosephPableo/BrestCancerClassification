import keras
import streamlit as st
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
model = load_model('./breastcancer_classification_model.h5')
min_value = np.float32(np.finfo(np.float32).min)  
max_value = np.float32(np.finfo(np.float32).max) 

st.title("Breast Cancer Classification")

st.write("""
    A breast cancer classification using the data from fine needle aspiration (FNA) biopsy
    """)

st.write("""
            Mean
         """)
radius_m = st.number_input('Enter the mean radius', min_value=min_value, max_value=max_value, )
texture_m = st.number_input('Enter the mean texture', min_value=min_value, max_value=max_value, )
perimeter_m = st.number_input('Enter the mean perimeter', min_value=min_value, max_value=max_value, )
area_m = st.number_input('Enter the mean area', min_value=min_value, max_value=max_value, )
smoothness_m = st.number_input('Enter the mean smoothness', min_value=min_value, max_value=max_value, )
compactness_m = st.number_input('Enter the mean compactness', min_value=min_value, max_value=max_value, )
concavity_m = st.number_input('Enter the mean concavity', min_value=min_value, max_value=max_value, )
concave_points_m = st.number_input('Enter the mean concave points', min_value=min_value, max_value=max_value, )
symmetry_m = st.number_input('Enter the mean symmetry', min_value=min_value, max_value=max_value, )
fractal_dimension_m = st.number_input('Enter the mean fractal dimension', min_value=min_value, max_value=max_value, )
st.write("""
            Standard Error
         """)
radius_se = st.number_input('Enter the standard error radius', min_value=min_value, max_value=max_value, )
texture_se = st.number_input('Enter the standard error texture', min_value=min_value, max_value=max_value, )
perimeter_se = st.number_input('Enter the standard error perimeter', min_value=min_value, max_value=max_value, )
area_se = st.number_input('Enter the standard error area', min_value=min_value, max_value=max_value, )
smoothness_se = st.number_input('Enter the standard error smoothness', min_value=min_value, max_value=max_value, )
compactness_se = st.number_input('Enter the standard error compactness', min_value=min_value, max_value=max_value, )
concavity_se = st.number_input('Enter the standard error concavity', min_value=min_value, max_value=max_value, )
concave_points_se = st.number_input('Enter a concave points', min_value=min_value, max_value=max_value, )
symmetry_se = st.number_input('Enter the standard error symmetry', min_value=min_value, max_value=max_value, )
fractal_dimension_se = st.number_input('Enter the standard error fractal dimension', min_value=min_value, max_value=max_value, )
st.write("""
            Worst
         """)
radius_w = st.number_input('Enter the worst mean', min_value=min_value, max_value=max_value, )
texture_w = st.number_input('Enter the worst texture', min_value=min_value, max_value=max_value, )
perimeter_w = st.number_input('Enter the worst perimeter', min_value=min_value, max_value=max_value, )
area_w = st.number_input('Enter the worst area', min_value=min_value, max_value=max_value, )
smoothness_w = st.number_input('Enter the worst smoothness', min_value=min_value, max_value=max_value, ) 
compactness_w = st.number_input('Enter the worst compactness', min_value=min_value, max_value=max_value, )
concavity_w = st.number_input('Enter the worst concavity', min_value=min_value, max_value=max_value, )
concave_points_w = st.number_input('Enter the worst concave points', min_value=min_value, max_value=max_value, )
symmetry_w = st.number_input('Enter the worst symmetry', min_value=min_value, max_value=max_value, )
fractal_dimension_w = st.number_input('Enter the worst fractal dimension', min_value=min_value, max_value=max_value, )

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
