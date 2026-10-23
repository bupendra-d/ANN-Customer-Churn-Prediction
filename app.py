import pandas as pd
import pickle

import tensorflow as tf
import streamlit as st


# Load Model and Preprocessors
model = tf.keras.models.load_model("model.h5")

with open("label_encoder.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# Streamlit User Interface
st.title("Customer Churn Prediction System")
st.write("Enter customer details to predict churn probability.")

geography = st.selectbox(
    "Geography", onehot_encoder_geo.categories_[0].tolist()
)
gender = st.selectbox(
    "Gender", label_encoder_gender.classes_.tolist()
)

age = st.slider("Age", 18, 92, 35)
credit_score = st.number_input(
    "Credit Score", min_value=300, max_value=850,
    value=650, step=1
)
tenure = st.slider("Tenure", 0, 10, 5)
balance = st.number_input(
    "Balance", min_value=0.0, value=0.0, step=100.0
)
num_of_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input(
    "Estimated Salary", min_value=0.0,
    value=50000.0, step=1000.0
)



# Prepare Input Features
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]])

if hasattr(geo_encoded, "toarray"):
    geo_encoded = geo_encoded.toarray()

geo_columns = onehot_encoder_geo.get_feature_names_out(["Geography"])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_columns)

# Combine features
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

expected_columns = list(scaler.feature_names_in_)
input_data.columns = input_data.columns.map(str)

input_data = input_data.rename(columns={
    col: str((col,))
    for col in geo_columns
    if str((col,)) in expected_columns
})

if set(input_data.columns) != set(expected_columns):
    st.error("Input features do not match the scaler's training features.")
    st.stop()

input_data = input_data[expected_columns]


# Scale Features and Predict
input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled, verbose=0)
churn_probability = float(prediction.ravel()[0])


# Display Prediction
st.subheader("Prediction Result")
st.metric("Churn Probability", f"{churn_probability:.2%}")

if churn_probability > 0.5:
    st.error("Customer is likely to churn.")
else:
    st.success("Customer is likely to stay.")
