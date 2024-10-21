import streamlit as st
import pickle
import numpy as np

# Load the model and scaler from the pickle files
with open('breast_cancer_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI
st.title("Breast Cancer Predictor")

# Input: 30 features (as in the notebook) for user input
input_features = []
for i in range(30):
    feature_value = st.number_input(f"Feature {i+1}", value=0.0)
    input_features.append(feature_value)

# Convert the input features into a NumPy array
input_data = np.array(input_features).reshape(1, -1)

# Normalize the input data using the scaler

# Predict the outcome (benign/malignant)
prediction = model.predict(normalized_input)

# Display the result
if st.button('Predict'):
    if prediction[0] == 1:
        st.write("The prediction is: Malignant")
    else:
        st.write("The prediction is: Benign")
