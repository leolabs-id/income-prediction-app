import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Income Prediction App",
    layout="centered"
)

st.title("💰 Income Prediction System")
st.write(
    "Aplikasi ini memprediksi apakah pendapatan seseorang "
    "**lebih dari $50K per tahun** berdasarkan data sensus "
    "menggunakan model **XGBoost**."
)

st.divider()

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_model():
    with open("model_income_xgboost.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Ambil nama fitur hasil training
feature_names = model.get_booster().feature_names

# ======================================================
# EDUCATION TABLE (Adult Census)
# ======================================================
education_map = {
    "Preschool": 1,
    "1st–4th": 2,
    "5th–6th": 3,
    "7th–8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16
}

# ======================================================
# INPUT USER
# ======================================================
st.subheader("📋 Profil Individu")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia", 17, 90, 30)
    hours_per_week = st.number_input("Jam Kerja per Minggu", 1, 100, 40)
    education_label = st.selectbox(
        "Pendidikan Terakhir",
        list(education_map.keys()),
        index=list(education_map.keys()).index("Bachelors")
    )

with col2:
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = st.number_input("Capital Loss", 0, 100000, 0)

education_num = education_map[education_label]

st.caption(
    f"Pendidikan **{education_label}** dikonversi ke EducationNum = {education_num}"
)

st.divider()

# ---------------- CATEGORICAL INPUT ----------------
workclass = st.selectbox(
    "Workclass",
    [
        "Private", "Self-emp-not-inc", "Self-emp-inc",
        "Federal-gov", "Local-gov", "State-gov",
        "Without-pay", "Never-worked"
    ]
)

marital_status = st.selectbox(
    "Status Pernikahan",
    [
        "Married-civ-spouse", "Never-married", "Divorced",
        "Separated", "Widowed", "Married-spouse-absent"
    ]
)

occupation = st.selectbox(
    "Occupation",
    [
        "Exec-managerial", "Prof-specialty", "Sales", "Tech-support",
        "Craft-repair", "Machine-op-inspct", "Adm-clerical",
        "Handlers-cleaners", "Other-service", "Transport-moving",
        "Farming-fishing", "Protective-serv", "Priv-house-serv",
        "Armed-Forces", "No-occupation"
    ]
)

relationship = st.selectbox(
    "Relationship",
    [
        "Husband", "Wife", "Own-child",
        "Not-in-family", "Other-relative", "Unmarried"
    ]
)

gender = st.radio("Jenis Kelamin", ["Male", "Female"])

st.divider()

# ======================================================
# PREDICTION
# ======================================================
if st.button("🔍 Prediksi Pendapatan"):
    try:
        # --------------------------------------------------
        # Buat dataframe kosong sesuai fitur training
        # --------------------------------------------------
        input_df = pd.DataFrame(
            data=np.zeros((1, len(feature_names))),
            columns=feature_names
        )

        # ---------------- NUMERICAL FEATURES -------------
        numeric_features = {
            "Age": age,
            "EducationNum": education_num,
            "Hours per Week": hours_per_week,
            "Capital Gain": capital_gain,
            "capital loss": capital_loss,
            "Has_Capital_Gain": int(capital_gain > 0),
            "Has_Capital_Loss": int(capital_loss > 0)
        }

        for col, val in numeric_features.items():
            if col in input_df.columns:
                input_df.at[0, col] = val

        # ---------------- ONE-HOT CATEGORICAL -------------
        def activate(col_name):
            if col_name in input_df.columns:
                input_df.at[0, col_name] = 1

        activate(f"Workclass_{workclass}")
        activate(f"Marital Status_{marital_status}")
        activate(f"Occupation_{occupation}")
        activate(f"Relationship_{relationship}")
        activate(f"Gender_{gender}")

        # ---------------- MODEL PREDICTION ----------------
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("📊 Hasil Prediksi")

        if prediction == 1:
            st.success("Prediksi: **Pendapatan > $50K per tahun**")
            st.metric("Probabilitas Pendapatan Tinggi", f"{probability*100:.2f}%")
        else:
            st.warning("Prediksi: **Pendapatan ≤ $50K per tahun**")
            st.metric("Probabilitas Pendapatan Rendah", f"{(1-probability)*100:.2f}%")

        st.info(
            f"""
            **Ringkasan Profil**
            - Usia: {age} tahun
            - Pendidikan: {education_label}
            - Jam Kerja: {hours_per_week} jam/minggu
            - Workclass: {workclass}
            - Marital Status: {marital_status}
            - Occupation: {occupation}
            - Relationship: {relationship}
            - Gender: {gender}
            """
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")