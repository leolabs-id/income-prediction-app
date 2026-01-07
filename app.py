import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Income AI Predictor",
    layout="wide"
)

# =====================================================
# 2. LOAD RESOURCES
# =====================================================
@st.cache_resource
def load_resources():
    with open("model_income_xgboost.pkl", "rb") as f:
        model = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    return model, explainer

model, explainer = load_resources()

# =====================================================
# 3. MAPPINGS (Indonesia Translation)
# =====================================================
EDUCATION_MAP = {
    "Tidak Sekolah": 1, "SD": 3, "SMP": 5, "SMA": 9, "Kuliah/Diploma": 10,
    "Sarjana (S1)": 13, "Magister (S2)": 14, "Pendidikan Profesi": 15, "Doktor (S3)": 16
}

OCCUPATION_MAP = {
    "Manajerial / Eksekutif": "Exec-managerial", "Profesional / Spesialis": "Prof-specialty",
    "Teknisi / IT Support": "Tech-support", "Penjualan": "Sales", "Layanan Umum": "Other-service",
    "Pekerja Kasar": "Handlers-cleaners", "Lainnya": "Unknown"
}

MARITAL_MAP = {"Belum Menikah": "Never-married", "Menikah": "Married-civ-spouse", "Cerai/Pisah": "Divorced"}
RELATIONSHIP_MAP = {"Suami": "Husband", "Istri": "Wife", "Anak": "Own-child", "Lainnya": "Unmarried"}

# =====================================================
# 4. INPUT PROFIL (SEMUA TERBUKA / STATIS)
# =====================================================
st.title("💰 Income Prediction & Business Insight")
st.subheader("📋 Input Profil Individu")

# Membagi input ke dalam 3 kolom agar semua terlihat tanpa scroll jauh
col_in1, col_in2, col_in3 = st.columns(3)

with col_in1:
    age = st.slider("Usia", 17, 90, 35)
    education_label = st.selectbox("Pendidikan Terakhir", list(EDUCATION_MAP.keys()), index=5)
    gender = st.radio("Jenis Kelamin", ["Male", "Female"], horizontal=True)

with col_in2:
    occupation_label = st.selectbox("Pekerjaan", list(OCCUPATION_MAP.keys()))
    marital_label = st.selectbox("Status Pernikahan", list(MARITAL_MAP.keys()))
    relationship_label = st.selectbox("Hubungan Keluarga", list(RELATIONSHIP_MAP.keys()))

with col_in3:
    hours = st.number_input("Jam Kerja per Minggu", 1, 100, 40)
    # Teks tambahan untuk rata-rata harian
    avg_daily = hours / 5
    st.markdown(f"**⏱️ Estimasi:** {avg_daily:.1f} jam / hari (5 hari kerja)")
    
    capital_gain = st.number_input("Capital Gain ($)", 0, 100000, 0)
    capital_loss = st.number_input("Capital Loss ($)", 0, 100000, 0)

st.divider()

# =====================================================
# 5. PREDICTION LOGIC
# =====================================================
def build_input():
    feature_names = model.get_booster().feature_names
    X = pd.DataFrame(0, index=[0], columns=feature_names)
    X["Age"] = age
    X["EducationNum"] = EDUCATION_MAP[education_label]
    X["Hours per Week"] = hours
    X["Capital Gain"] = capital_gain
    X["capital loss"] = capital_loss
    
    # Aktivasi One-Hot
    def act(prefix, val):
        col = f"{prefix}_{val}"
        if col in X.columns: X[col] = 1

    act("Occupation", OCCUPATION_MAP[occupation_label])
    act("Marital Status", MARITAL_MAP[marital_label])
    act("Relationship", RELATIONSHIP_MAP[relationship_label])
    if gender == "Male": X["Gender_Male"] = 1
    return X

X_input = build_input()

if st.button("🚀 Jalankan Analisis Prediksi", use_container_width=True):
    prob = model.predict_proba(X_input)[0][1]
    pred = 1 if prob > 0.5 else 0

    # --- BAGIAN HASIL & GRAFIK ---
    col_res1, col_res2 = st.columns([1, 1.5])

    with col_res1:
        st.subheader("🎯 Hasil Prediksi")
        if pred == 1:
            st.success("### PENDAPATAN TINGGI (> $50K)")
            label = "Tinggi"
        else:
            st.error("### PENDAPATAN RENDAH (≤ $50K)")
            label = "Rendah"
            prob = 1 - prob
        
        st.metric(f"Confidence (Prob. {label})", f"{prob*100:.2f}%")

    with col_res2:
        st.subheader("🔎 Penjelasan Model (SHAP)")
        shap_values = explainer.shap_values(X_input)[0]
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plotting manual untuk kontrol warna lebih baik
        shap_df = pd.DataFrame({"Fitur": X_input.columns, "Impact": shap_values})
        shap_df = shap_df.sort_values("Impact", key=abs).tail(10)
        colors = ['#ff4b4b' if x < 0 else '#00cc96' for x in shap_df['Impact']]
        
        ax.barh(shap_df['Fitur'], shap_df['Impact'], color=colors)
        ax.set_title("10 Faktor Utama Penentu Prediksi")
        st.pyplot(fig)

    # --- TAFSIR ANALISIS DESKRIPTIF BISNIS ---
    st.divider()
    st.subheader("📝 Tafsir Analisis & Rekomendasi Bisnis")
    
    col_bis1, col_bis2 = st.columns(2)
    
    with col_bis1:
        st.markdown("#### 🧐 Mengapa Prediksi Demikian?")
        if pred == 0:
            st.write(f"Model mendeteksi bahwa tingkat pendidikan **{education_label}** dikombinasikan dengan jenis pekerjaan **{occupation_label}** secara historis memiliki batas atas pendapatan di bawah $50K.")
            st.write(f"Meskipun bekerja **{avg_daily:.1f} jam/hari**, faktor kualifikasi (EducationNum) memberikan dampak negatif yang lebih besar pada probabilitas.")
        else:
            st.write(f"Kombinasi usia matang (**{age} tahun**) dan sektor pekerjaan **{occupation_label}** merupakan pendorong utama.")
            st.write(f"Status **{marital_label}** juga terdeteksi oleh model sebagai faktor stabilitas yang berkorelasi dengan pendapatan tinggi di dataset ini.")

    with col_bis2:
        st.markdown("#### 🚀 Strategi Tindak Lanjut")
        if pred == 0:
            st.warning("**Rekomendasi:** Fokus pada peningkatan spesialisasi. Data menunjukkan transisi ke pendidikan 'Sarjana' atau 'Magister' akan menggeser dampak fitur merah menjadi hijau secara signifikan.")
        else:
            st.info("**Rekomendasi:** Pertahankan posisi di industri saat ini. Manfaatkan 'Capital Gain' untuk diversifikasi aset karena profil Anda berada pada segmen profitabilitas tinggi.")