import streamlit as st
import requests

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Prediction")
st.write("Enter patient details and click **Predict**. This UI calls the FastAPI backend.")

# Local default. In Docker later, use http://api:8000
API_URL = st.sidebar.text_input("FastAPI URL", value="http://127.0.0.1:8000")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("age", min_value=0, max_value=120, value=50)
    sex = st.selectbox("sex (0=female, 1=male)", [0, 1], index=1)
    cp = st.number_input("cp", min_value=0, max_value=3, value=0)
    trestbps = st.number_input("trestbps", min_value=0, max_value=300, value=120)

with col2:
    chol = st.number_input("chol", min_value=0, max_value=700, value=200)
    fbs = st.selectbox("fbs (0/1)", [0, 1], index=0)
    restecg = st.number_input("restecg", min_value=0, max_value=2, value=0)
    thalach = st.number_input("thalach", min_value=0, max_value=250, value=150)

with col3:
    exang = st.selectbox("exang (0/1)", [0, 1], index=0)
    oldpeak = st.number_input("oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.number_input("slope", min_value=0, max_value=2, value=1)
    ca = st.number_input("ca", min_value=0, max_value=4, value=0)
    thal = st.number_input("thal", min_value=0, max_value=3, value=1)

payload = {
    "age": int(age),
    "sex": int(sex),
    "cp": int(cp),
    "trestbps": int(trestbps),
    "chol": int(chol),
    "fbs": int(fbs),
    "restecg": int(restecg),
    "thalach": int(thalach),
    "exang": int(exang),
    "oldpeak": float(oldpeak),
    "slope": int(slope),
    "ca": int(ca),
    "thal": int(thal),
}

if st.button("Predict"):
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        if r.status_code != 200:
            st.error(f"API error {r.status_code}: {r.text}")
        else:
            out = r.json()
            pred = out.get("prediction")
            prob = out.get("probability", 0.0)

            if pred == 1:
                st.error(f"Prediction: HEART DISEASE (1) — Probability: {prob:.3f}")
            elif pred == 0:
                st.success(f"Prediction: NO HEART DISEASE (0) — Probability: {prob:.3f}")
            else:
                st.warning(f"Unexpected response: {out}")

            st.json(out)

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI. Start the API first on port 8000.")
    except Exception as e:
        st.error(f"Error: {e}")
