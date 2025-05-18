
import streamlit as st
import pandas as pd
import joblib

# Load saved model components
mlp = joblib.load("mlp_model.pkl")
pca = joblib.load("pca_transform.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Streamlit GUI Title
st.title("🧠 Intelligent Donor Recommendation for Liver Transplantation")
st.markdown("A PCA + MLP based system for predicting survival and recommending optimal donors.")

# User Input Form
with st.form("Donor Form"):
    donor_age = st.slider("Donor Age", 18, 70, 45)
    recipient_age = st.slider("Recipient Age", 18, 65, 50)
    bilirubin = st.number_input("Bilirubin Level", min_value=0.1, max_value=10.0, value=1.5)
    INR = st.number_input("INR", min_value=0.5, max_value=5.0, value=1.0)
    creatinine = st.number_input("Creatinine Level", min_value=0.1, max_value=10.0, value=1.2)
    cold_ischemia_time = st.slider("Cold Ischemia Time (hrs)", 2, 12, 6)
    gender = st.selectbox("Recipient Gender", ["Male", "Female"])
    match_score = st.slider("Donor-Recipient Match Score", 50, 100, 85)
    submit = st.form_submit_button("Predict Outcome")

# Prediction Function
def predict(data_dict):
    df_input = pd.DataFrame([data_dict])
    df_input['gender'] = label_encoder.transform(df_input['gender'])
    scaled_input = scaler.transform(df_input)
    pca_input = pca.transform(scaled_input)
    prediction = mlp.predict(pca_input)[0]
    return "✅ Donor is Suitable" if prediction == 1 else "❌ Donor Not Suitable"

# Display Prediction
if submit:
    user_data = {
        "donor_age": donor_age,
        "recipient_age": recipient_age,
        "bilirubin": bilirubin,
        "INR": INR,
        "creatinine": creatinine,
        "cold_ischemia_time": cold_ischemia_time,
        "gender": gender,
        "match_score": match_score
    }
    result = predict(user_data)
    st.success(result)
