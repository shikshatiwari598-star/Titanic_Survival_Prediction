import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("titanic_survival_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# App Title
st.title("🚢 Titanic Survival Prediction")
st.write("Predict whether a passenger would survive or not.")

st.divider()

# Passenger Details
st.subheader("Enter Passenger Details")

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

gender = st.selectbox(
    "Gender",
    ["male", "female"]
)

age = st.slider(
    "Age",
    1, 80, 25
)

sibsp = st.number_input(
    "Number of Siblings/Spouses",
    min_value=0,
    max_value=8,
    value=0
)

parch = st.number_input(
    "Number of Parents/Children",
    min_value=0,
    max_value=6,
    value=0
)

fare = st.number_input(
    "Ticket Fare",
    min_value=0.0,
    max_value=600.0,
    value=50.0
)

embarked = st.selectbox(
    "Embarked",
    ["S", "C", "Q"]
)

st.divider()

# Prediction
if st.button("Predict Survival", use_container_width=True):

    new_passenger = pd.DataFrame({
        "Pclass": [pclass],
        "Gender": [gender],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })

    prediction = model.predict(new_passenger)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Passenger is predicted to SURVIVE! 🎉")
    else:
        st.error("Passenger is predicted to NOT SURVIVE.")

    st.write("Prediction Class:", prediction)

st.divider()

st.caption("Titanic Survival Prediction | Logistic Regression")
