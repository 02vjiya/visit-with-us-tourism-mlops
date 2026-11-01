
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Visit with Us | Wellness Package", page_icon="✈️", layout="wide")
st.title("Visit with Us — Wellness Tourism Package")
st.write("Estimate whether a customer is likely to purchase the Wellness Tourism Package.")

MODEL_PATH = Path(__file__).with_name("tourism_model.joblib")
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

with st.form("customer_details"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
        city_tier = st.selectbox("City tier", [1, 2, 3], index=1)
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital status", ["Married", "Single", "Divorced"])
    with c2:
        duration = st.number_input("Duration of pitch (minutes)", min_value=0.0, value=10.0)
        persons = st.number_input("Number of persons visiting", min_value=1, max_value=10, value=2)
        followups = st.number_input("Number of follow-ups", min_value=0, max_value=20, value=3)
        product = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        property_star = st.selectbox("Preferred property star", [3, 4, 5], index=0)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    with c3:
        trips = st.number_input("Number of trips per year", min_value=0.0, value=3.0)
        passport = st.selectbox("Has passport?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        pitch_score = st.selectbox("Pitch satisfaction score", [1, 2, 3, 4, 5], index=2)
        own_car = st.selectbox("Owns a car?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        children = st.number_input("Children below age 5 visiting", min_value=0, max_value=10, value=0)
        income = st.number_input("Monthly income", min_value=0.0, value=25000.0)
    submitted = st.form_submit_button("Predict purchase likelihood")

if submitted:
    input_df = pd.DataFrame([{
        "Age": age, "TypeofContact": contact, "CityTier": city_tier,
        "DurationOfPitch": duration, "Occupation": occupation, "Gender": gender,
        "NumberOfPersonVisiting": persons, "NumberOfFollowups": followups,
        "ProductPitched": product, "PreferredPropertyStar": property_star,
        "MaritalStatus": marital, "NumberOfTrips": trips, "Passport": passport,
        "PitchSatisfactionScore": pitch_score, "OwnCar": own_car,
        "NumberOfChildrenVisiting": children, "Designation": designation,
        "MonthlyIncome": income
    }])
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0, 1])
    st.subheader("Prediction")
    st.metric("Purchase probability", f"{probability:.1%}")
    if prediction == 1:
        st.success("Likely to purchase — consider prioritizing this customer for follow-up.")
    else:
        st.info("Less likely to purchase — consider a tailored engagement strategy.")
    st.caption("This is a model estimate, not a guarantee of customer behavior.")
