import keras
import streamlit as st
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
model = load_model('./breastcancer_classification_model.h5')

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
min_value = np.float32(np.finfo(np.float32).min)  
max_value = np.float32(np.finfo(np.float32).max) 

st.title("Breast Cancer Classification")

st.write("""
    A breast cancer classification using the data from fine needle aspiration (FNA) biopsy
    """)

st.write("""
            Mean
         """)
radius_m = st.number_input('Enter the mean radius',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=11.76)
texture_m = st.number_input('Enter the mean texture',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=21.6)
perimeter_m = st.number_input('Enter the mean perimeter',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=74.72)
area_m = st.number_input('Enter the mean area',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=427.90)
smoothness_m = st.number_input('Enter the mean smoothness',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=0.086370)
compactness_m = st.number_input('Enter the mean compactness',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=0.049660)
concavity_m = st.number_input('Enter the mean concavity',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=0.016570)
concave_points_m = st.number_input('Enter the mean concave points',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=0.011150)
symmetry_m = st.number_input('Enter the mean symmetry',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=0.149500)
fractal_dimension_m = st.number_input('Enter the mean fractal dimension',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=0.058880)
st.write("""
            Standard Error
         """)
radius_se = st.number_input('Enter the standard error radius',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=0.406200)
texture_se = st.number_input('Enter the standard error texture',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=1.21)
perimeter_se = st.number_input('Enter the standard error perimeter',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=2.635)
area_se = st.number_input('Enter the standard error area',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=28.47)
smoothness_se = st.number_input('Enter the standard error smoothness',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=0.005857)
compactness_se = st.number_input('Enter the standard error compactness',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=0.009758)
concavity_se = st.number_input('Enter the standard error concavity',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=0.011680)
concave_points_se = st.number_input('Enter a concave points',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=0.007445)
symmetry_se = st.number_input('Enter the standard error symmetry',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=0.024060)
fractal_dimension_se = st.number_input('Enter the standard error fractal dimension',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=0.001769)
st.write("""
            Worst
         """)
radius_w = st.number_input('Enter the worst mean',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=12.98)
texture_w = st.number_input('Enter the worst texture',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=25.72)
perimeter_w = st.number_input('Enter the worst perimeter',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=82.98)
area_w = st.number_input('Enter the worst area',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=516.50)
smoothness_w = st.number_input('Enter the worst smoothness',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=0.10850) 
compactness_w = st.number_input('Enter the worst compactness',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=0.086150)
concavity_w = st.number_input('Enter the worst concavity',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=0.055230)
concave_points_w = st.number_input('Enter the worst concave points',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=0.037150)
symmetry_w = st.number_input('Enter the worst symmetry',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=0.243300)
fractal_dimension_w = st.number_input('Enter the worst fractal dimension',min_value=float(min_value), max_value=float(max_value), step=1e-6, format="%.6f", value=0.065630)

input = np.array([radius_m,texture_m,perimeter_m,area_m,smoothness_m,compactness_m,concavity_m,concave_points_m,symmetry_m,fractal_dimension_m,
                 radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,
                 radius_w,texture_w,perimeter_w,area_w,smoothness_w,compactness_w,concavity_w,concave_points_w,symmetry_w,fractal_dimension_w]) 
input_data_reshaped = input.reshape(1,-1)

if st.button('Classify'):
    prediction = model.predict(input_data_reshaped)
    st.write(prediction)
    prediction_label = [np.argmax(prediction)]
    if(prediction_label == 0):
      st.write('The tumor is Malignant')
    else:
      st.write('The tumor is Benign')

