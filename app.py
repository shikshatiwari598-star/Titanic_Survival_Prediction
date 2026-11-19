import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Titanic | Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0b132b, #172554, #101827);
    color: white;
}

.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

.hero {
    background: linear-gradient(120deg, #172554, #263b69);
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    border: 1px solid #64748b;
    margin-bottom: 30px;
}

.hero h1 {
    color: #fbbf24;
    font-size: 42px;
    margin-bottom: 10px;
}

.hero p {
    color: #e2e8f0;
    font-size: 17px;
}

.section {
    background: #172554;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #334b78;
    margin-bottom: 20px;
}

.section h3 {
    color: #fbbf24;
}

label, .stMarkdown p {
    color: #f1f5f9 !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #243653;
    border-radius: 10px;
}

.stButton > button {
    background: linear-gradient(90deg, #d69e24, #fbbf24);
    color: #111827;
    border: none;
    border-radius: 12px;
    height: 52px;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background: #fde68a;
    color: #111827;
}

.result-card {
    background: linear-gradient(120deg, #1e3a5f, #263b69);
    padding: 28px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #64748b;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: #94a3b8;
    padding: 25px;
    font-size: 13px;
}

/* Remove white header background */
[data-testid="stHeader"] {
    background: #0b132b !important;
}

/* Match toolbar with app background */
[data-testid="stToolbar"] {
    background: transparent !important;
}

/* Reduce top empty space */
.block-container {
    padding-top: 1rem !important;
}

/* Main application background */
[data-testid="stAppViewContainer"] {
    background: #0b132b;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return joblib.load("titanic_survival_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>🚢 TITANIC SURVIVAL PREDICTOR</h1>
    <p>Discover the prediction of passenger survival using Machine Learning</p>
    <p>Powered by Logistic Regression</p>
</div>
""", unsafe_allow_html=True)


# ---------------- PASSENGER FORM ----------------

st.markdown("""
<div class="section">
<h3>👤 Passenger Information</h3>
<p>Enter passenger details to predict the survival result.</p>
</div>
""", unsafe_allow_html=True)

with st.form("passenger_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Details")

        pclass = st.selectbox(
            "Passenger Class",
            [1, 2, 3]
        )

        gender = st.selectbox(
            "Gender",
            ["female", "male"]
        )

        age = st.slider(
            "Age",
            0, 80, 25
        )

        embarked = st.selectbox(
            "Embarked Location",
            ["S", "C", "Q"]
        )

    with col2:
        st.subheader("Travel Details")

        sibsp = st.number_input(
            "Siblings / Spouses",
            min_value=0,
            max_value=8,
            value=0
        )

        parch = st.number_input(
            "Parents / Children",
            min_value=0,
            max_value=6,
            value=0
        )

        fare = st.number_input(
            "Ticket Fare",
            min_value=0.0,
            max_value=600.0,
            value=50.0,
            step=1.0
        )

    st.markdown("---")

    submit = st.form_submit_button(
        "🚀 Predict Survival"
    )


# ---------------- PREDICTION ----------------

if submit:

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

    probability = model.predict_proba(new_passenger)[0]

    survival_probability = probability[1] * 100
    not_survival_probability = probability[0] * 100

    st.markdown(
        '<div class="result-card"><h2>📊 Prediction Result</h2></div>',
        unsafe_allow_html=True
    )

    if prediction == 1:
        st.success("🎉 Passenger is predicted to SURVIVE!")
    else:
        st.error("Passenger is predicted to NOT SURVIVE.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Survival Probability",
            f"{survival_probability:.2f}%"
        )

    with col2:
        st.metric(
            "Non-Survival Probability",
            f"{not_survival_probability:.2f}%"
        )

    st.progress(float(survival_probability / 100))


# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    TITANIC SURVIVAL PREDICTION |
    Machine Learning Project |
    Logistic Regression
</div>
""", unsafe_allow_html=True)
