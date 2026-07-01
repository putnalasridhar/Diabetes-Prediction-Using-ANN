import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>

.stApp{
    background-color:#0E1117;
    color:white;
}

h1,h2,h3,h4,h5,h6,label{
    color:white !important;
}

[data-testid="stSidebar"]{
    background:#161B22;
}

div[data-baseweb="select"]{
    color:black;
}

.stButton>button{
    background:#00C853;
    color:white;
    font-size:18px;
    border-radius:10px;
    height:55px;
    width:100%;
}

.stButton>button:hover{
    background:#00A844;
}

.block-container{
    padding-top:1rem;
    padding-bottom:2rem;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_files():
    model = load_model("diabetes_model.keras", compile=False)
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_files()
except Exception as e:
    st.error(f"Error loading model:\n{e}")
    st.stop()
# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("Diabetes Prediction System")

st.write(
    "Fill in the patient details below and click **Predict Diabetes**."
)

st.divider()

# --------------------------------------------------
# Three Columns
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

# ==================================================
# COLUMN 1
# ==================================================

with col1:

    st.subheader("Medical History")

    highbp = st.selectbox(
        "High Blood Pressure",
        ["No","Yes"]
    )

    highchol = st.selectbox(
        "High Cholesterol",
        ["No","Yes"]
    )

    cholcheck = st.selectbox(
        "Cholesterol Check Done",
        ["Yes","No"]
    )

    smoker = st.selectbox(
        "Smoker",
        ["No","Yes"]
    )

    stroke = st.selectbox(
        "Stroke",
        ["No","Yes"]
    )

    heart = st.selectbox(
        "Heart Disease / Heart Attack",
        ["No","Yes"]
    )

    healthcare = st.selectbox(
        "Any Healthcare",
        ["Yes","No"]
    )

    nodoc = st.selectbox(
        "Couldn't Afford Doctor",
        ["No","Yes"]
    )

    diffwalk = st.selectbox(
        "Difficulty Walking",
        ["No","Yes"]
    )

    genhlth = st.slider(
        "General Health",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Excellent, 5 = Poor"
    )
    # ==================================================
# COLUMN 2
# ==================================================

with col2:

    st.subheader("Lifestyle")

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=80.0,
        value=22.0,
        step=0.1
    )

    activity = st.selectbox(
        "Physical Activity",
        ["Yes", "No"]
    )

    fruits = st.selectbox(
        "Eat Fruits Regularly",
        ["Yes", "No"]
    )

    veggies = st.selectbox(
        "Eat Vegetables Regularly",
        ["Yes", "No"]
    )

    alcohol = st.selectbox(
        "Heavy Alcohol Consumption",
        ["No", "Yes"]
    )

    menthlth = st.slider(
        "Mental Health (Days)",
        min_value=0,
        max_value=30,
        value=0,
        help="Number of days mental health was not good in last 30 days."
    )

    physhlth = st.slider(
        "Physical Health (Days)",
        min_value=0,
        max_value=30,
        value=0,
        help="Number of days physical health was not good in last 30 days."
    )

# ==================================================
# COLUMN 3
# ==================================================

with col3:

    st.subheader("Personal Details")

    sex = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

age_options = {
    "18-24": 1,
    "25-29": 2,
    "30-34": 3,
    "35-39": 4,
    "40-44": 5,
    "45-49": 6,
    "50-54": 7,
    "55-59": 8,
    "60-64": 9,
    "65-69": 10,
    "70-74": 11,
    "75-79": 12,
    "80+": 13
}

selected_age = st.selectbox(
    "Age Group",
    list(age_options.keys())
)

age = age_options[selected_age]

education = st.selectbox(
        "Education Level",
        [1,2,3,4,5,6],
        help="""
1 = Never attended school
6 = College graduate
"""
    )

income = st.selectbox(
        "Income Level",
        [1,2,3,4,5,6,7,8],
        help="""
1 = Lowest Income
8 = Highest Income
"""
    )

st.divider()

# ==================================================
# Convert Yes / No to 1 / 0
# ==================================================

highbp = 1 if highbp == "Yes" else 0
highchol = 1 if highchol == "Yes" else 0
cholcheck = 1 if cholcheck == "Yes" else 0
smoker = 1 if smoker == "Yes" else 0
stroke = 1 if stroke == "Yes" else 0
heart = 1 if heart == "Yes" else 0
activity = 1 if activity == "Yes" else 0
fruits = 1 if fruits == "Yes" else 0
veggies = 1 if veggies == "Yes" else 0
alcohol = 1 if alcohol == "Yes" else 0
healthcare = 1 if healthcare == "Yes" else 0
nodoc = 1 if nodoc == "Yes" else 0
diffwalk = 1 if diffwalk == "Yes" else 0
sex = 1 if sex == "Male" else 0
# ==================================================
# Predict Button
# ==================================================

if st.button("Predict Diabetes", use_container_width=True):

    # Create DataFrame in the SAME ORDER as training
    user_data = pd.DataFrame({
        'HighBP': [highbp],
        'HighChol': [highchol],
        'CholCheck': [cholcheck],
        'BMI': [bmi],
        'Smoker': [smoker],
        'Stroke': [stroke],
        'HeartDiseaseorAttack': [heart],
        'PhysActivity': [activity],
        'Fruits': [fruits],
        'Veggies': [veggies],
        'HvyAlcoholConsump': [alcohol],
        'AnyHealthcare': [healthcare],
        'NoDocbcCost': [nodoc],
        'GenHlth': [genhlth],
        'MentHlth': [menthlth],
        'PhysHlth': [physhlth],
        'DiffWalk': [diffwalk],
        'Sex': [sex],
        'Age': [age],
        'Education': [education],
        'Income': [income]
    })

    # Scale data
    scaled_data = scaler.transform(user_data)

    # Prediction
    probability = float(model.predict(scaled_data, verbose=0)[0][0])

    prediction = 1 if probability >= 0.5 else 0

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("High Risk of Diabetes")

        st.progress(min(int(probability * 100), 100))

        st.metric(
            label="Probability",
            value=f"{probability*100:.2f}%"
        )

        st.info("""
### Recommendations

✔ Visit a healthcare professional.

✔ Maintain a healthy diet.

✔ Exercise regularly.

✔ Monitor blood sugar levels.

✔ Avoid smoking and alcohol.
""")

    else:

        st.success("Low Risk of Diabetes")

        st.progress(min(int(probability * 100), 100))

        st.metric(
            label="Probability",
            value=f"{probability*100:.2f}%"
        )

        st.info("""
### Recommendations

✔ Continue regular exercise.

✔ Eat fruits and vegetables.

✔ Maintain healthy body weight.

✔ Annual health check-up.

✔ Continue healthy lifestyle.
""")

    st.divider()

    with st.expander("View Entered Details"):

        st.dataframe(
            user_data,
            use_container_width=True
        )

# ==================================================
# Footer
# ==================================================

st.markdown("---")

st.markdown(
    """
<div style='text-align:center;color:gray;'>

Developed using Streamlit & TensorFlow

</div>
""",
unsafe_allow_html=True
)