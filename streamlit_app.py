import streamlit as st

st.title("Breast Cancer Classification")

st.write("""
    A breast cancer classification using the data from fine needle aspiration (FNA) biopsy
    """)

st.write("""
            Mean
         """)
radius_m = st.number_input('Enter the mean radius')
texture_m = st.number_input('Enter the mean texture')
perimeter_m = st.number_input('Enter the mean perimeter')
area_m = st.number_input('Enter the mean area')
smoothness_m = st.number_input('Enter the mean smoothness')
compactness_m = st.number_input('Enter the mean compactness')
concavity_m = st.number_input('Enter the mean concavity')
concave_points_m = st.number_input('Enter the mean concave points')
symmetry_m = st.number_input('Enter the mean symmetry')
fractal_dimension_m = st.number_input('Enter the mean fractal dimension')
st.write("""
            Standard Error
         """)
radius_se = st.number_input('Enter the standard error radius')
texture_se = st.number_input('Enter the standard error texture')
perimeter_se = st.number_input('Enter the standard error perimeter')
area_se = st.number_input('Enter the standard error area')
smoothness_se = st.number_input('Enter the standard error smoothness')
compactness_se = st.number_input('Enter the standard error compactness')
concavity_se = st.number_input('Enter the standard error concavity')
concave_points_se = st.number_input('Enter a concave points')
symmetry_se = st.number_input('Enter the standard error symmetry')
fractal_dimension_se = st.number_input('Enter the standard error fractal dimension')
st.write("""
            Worst
         """)
radius_w = st.number_input('Enter the worst mean')
texture_w = st.number_input('Enter the worst texture')
perimeter_w = st.number_input('Enter the worst perimeter')
area_w = st.number_input('Enter the worst area')
smoothness_w = st.number_input('Enter the worst smoothness') 
compactness_w = st.number_input('Enter the worst compactness')
concavity_w = st.number_input('Enter the worst concavity')
concave_points_w = st.number_input('Enter the worst concave points')
symmetry_w = st.number_input('Enter the worst symmetry')
fractal_dimension_w = st.number_input('Enter the worst fractal dimension')

input = np.array[radius_m,texture_m,perimeter_m,area_m,smoothness_m,compactness_m,concavity_m,concave_points_m,symmetry_m,fractal_dimension_m,
                 radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,
                 radius_w,texture_w,perimeter_w,area_w,smoothness_w,compactness_w,concavity_w,concave_points_w,symmetry_w,fractal_dimension_w] 

if st.button('Classify'):
    st.write(input)
else:
    st.write('Goodbye')


