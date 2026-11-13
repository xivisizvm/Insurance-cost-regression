# Insurance Cost Predictor - Streamlit demo app
# Run with:  streamlit run app/app.py   (from the repo root)

import os
import joblib
import pandas as pd
import streamlit as st

# ---------- Load the saved model and scaler ----------
# model.pkl and scaler.pkl sit in the repo root, one folder above /app
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@st.cache_resource
def load_files():
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    return model, scaler


model, scaler = load_files()


# ---------- Prediction function ----------
def build_features(age, sex, bmi, children, smoker, region):
    """Turn raw user input into the exact 9 columns the model was trained on."""
    # 1. Scale age, bmi, children (same order the scaler was fitted on)
    num = pd.DataFrame([[age, bmi, children]], columns=["age", "bmi", "children"])
    scaled = scaler.transform(num)
    age_s, bmi_s, children_s = scaled[0]

    # 2. Encode the categories the same way as preprocessing
    sex_val = 1 if sex == "Male" else 0
    smoker_val = 1 if smoker == "Yes" else 0

    # 3. Region dummies (northeast is the baseline, so all zeros)
    row = {
        "age": age_s,
        "sex": sex_val,
        "bmi": bmi_s,
        "children": children_s,
        "smoker": smoker_val,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }

    # 4. Interaction term uses the SCALED bmi (same as in the modeling notebook)
    row["bmi_smoker"] = bmi_s * smoker_val

    return pd.DataFrame([row])


def predict_charge(age, sex, bmi, children, smoker, region):
    features = build_features(age, sex, bmi, children, smoker, region)
    return float(model.predict(features)[0]), features


# ---------- Page layout ----------
st.set_page_config(page_title="Insurance Cost Predictor", page_icon="💰")
st.title("Medical Insurance Cost Predictor")
st.write("Enter your details to get an estimated yearly insurance charge.")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 64, 30)
    bmi = st.slider("BMI", 15.0, 54.0, 28.0, step=0.1)
    children = st.slider("Number of children", 0, 5, 0)
with col2:
    sex = st.selectbox("Sex", ["Male", "Female"])
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

show_quit = st.checkbox("What if this person quits smoking?")

# ---------- Predict ----------
if st.button("Predict"):
    charge, features = predict_charge(age, sex, bmi, children, smoker, region)
    charge = max(charge, 0)  # a linear model can go slightly negative for very low-risk inputs

    st.subheader("Estimated yearly charge")
    st.metric("Predicted cost", f"${charge:,.2f}")

    # Stretch goal: what-if quit smoking
    if show_quit:
        if smoker == "Yes":
            new_charge, _ = predict_charge(age, sex, bmi, children, "No", region)
            new_charge = max(new_charge, 0)
            st.success(
                f"If they quit smoking: ${new_charge:,.2f}  "
                f"(saves about ${charge - new_charge:,.2f} per year)"
            )
        else:
            st.info("This person is already a non-smoker, so there is nothing to compare.")

    # Which factors pushed the prediction up or down (coefficient x feature value)
    st.subheader("What influenced this prediction?")
    contributions = pd.Series(model.coef_ * features.iloc[0].values, index=features.columns)
    st.caption(
        "Each bar shows how much a feature moved the estimate, in dollars. "
        "Features are scaled, so this shows effect relative to an average person."
    )
    st.bar_chart(contributions.sort_values())

st.caption("Model: linear regression with a BMI x smoker interaction term. For educational use only.")
