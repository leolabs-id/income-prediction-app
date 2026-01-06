import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Income Prediction App",
    layout="centered"
)

st.title("💰 Income Prediction System")

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_model():
    with open("model_income_xgboost.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
feature_names = model.get_booster().feature_names

# ======================================================
# MAPPING PENDIDIKAN
# ======================================================
education_map = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4,
    "9th": 5, "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9,
    "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
    "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16
}

# ======================================================
# INPUT USER
# ======================================================
st.subheader("📋 Profil Individu")

# Baris 1: Pekerjaan & Status
col_a, col_b = st.columns(2)
with col_a:
    workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov"])
    marital_status = st.selectbox("Status Pernikahan", ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed", "Married-spouse-absent"])
with col_b:
    occupation = st.selectbox("Occupation", ["Exec-managerial", "Prof-specialty", "Sales", "Tech-support", "Craft-repair", "Adm-clerical", "Other-service"])
    relationship = st.selectbox("Relationship", ["Husband", "Wife", "Own-child", "Not-in-family", "Unmarried"])

# Baris 2: Pendidikan (Dropdown Baru) & Gender
col_c, col_d = st.columns(2)
with col_c:
    education_label = st.selectbox("Pendidikan Terakhir", list(education_map.keys()), index=12) # Default Bachelors
    education_num = education_map[education_label] # Konversi otomatis ke angka
with col_d:
    gender = st.radio("Jenis Kelamin", ["Male", "Female"], horizontal=True)

# Fitur numerik lainnya dalam expander
with st.expander("Fitur Tambahan (Umur, Jam Kerja, & Capital)"):
    age = st.slider("Usia", 17, 90, 30)
    hours_per_week = st.number_input("Jam Kerja per Minggu", 1, 100, 40)
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = st.number_input("Capital Loss", 0, 100000, 0)

st.divider()

# ======================================================
# PREDICTION LOGIC
# ======================================================
if st.button("📊 Prediksi Pendapatan"):
    try:
        # Menyiapkan DataFrame Input
        input_df = pd.DataFrame(data=np.zeros((1, len(feature_names))), columns=feature_names)
        
        # Isi Fitur Numerik
        input_df.at[0, "Age"] = age
        input_df.at[0, "EducationNum"] = education_num
        input_df.at[0, "Hours per Week"] = hours_per_week
        input_df.at[0, "Capital Gain"] = capital_gain
        input_df.at[0, "capital loss"] = capital_loss

        # Isi Fitur Kategorikal (One-Hot)
        def activate(col_name):
            if col_name in input_df.columns: input_df.at[0, col_name] = 1

        activate(f"Workclass_{workclass}")
        activate(f"Marital Status_{marital_status}")
        activate(f"Occupation_{occupation}")
        activate(f"Relationship_{relationship}")
        activate(f"Gender_{gender}")

        # Prediksi
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # --- TAMPILAN HASIL PREDIKSI ---
        st.subheader("📉 Hasil Prediksi")
        
        if prediction == 1:
            st.success(f"**Prediksi: Pendapatan > $50K per tahun**")
            prob_val = probability * 100
            label = "Tinggi"
        else:
            st.error(f"**Prediksi: Pendapatan ≤ $50K per tahun**")
            prob_val = (1 - probability) * 100
            label = "Rendah"

        st.metric(f"Probabilitas Pendapatan {label}", f"{prob_val:.2f}%")

        # --- SHAP VALUE VISUALIZATION ---
        st.divider()
        st.subheader("🔎 Penjelasan Prediksi (SHAP Value)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        fig, ax = plt.subplots()
        shap.bar_plot(shap_values[0], max_display=10, feature_names=feature_names, show=False)
        st.pyplot(fig)

        # --- TAFSIR DESKRIPTIF & SARAN ---
        st.divider()
        st.subheader("📝 Tafsir & Saran")
        
        if prediction == 0:
            st.write(f"### Mengapa Hasilnya Pendapatan Rendah?")
            st.write(f"Berdasarkan data, tingkat pendidikan **{education_label}** dan jam kerja **{hours_per_week} jam** menjadi faktor penahan.")
            st.info("💡 **Saran:** Cobalah tingkatkan skill spesifik atau sertifikasi untuk menaikkan posisi tawar Anda ke level 'Masters' atau 'Prof-school'.")
        else:
            st.write(f"### Mengapa Hasilnya Pendapatan Tinggi?")
            st.write(f"Kombinasi pendidikan **{education_label}** dan peran **{occupation}** sangat kuat mendorong pendapatan Anda ke atas.")
            st.info("💡 **Saran:** Pertahankan performa dan kelola Capital Gain Anda agar aset terus tumbuh.")

    except Exception as e:
        st.error(f"Error: {e}")