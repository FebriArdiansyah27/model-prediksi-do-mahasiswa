import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# KONFIGURASI DASHBOARD
# -------------------------------
st.set_page_config(page_title="Dashboard Prediksi DO Mahasiswa", layout="wide")

# -------------------------------
# LOAD DATASET & MODEL
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv('dataset_risiko_do_mahasiswa1500.csv')

@st.cache_resource
def load_model():
    model = joblib.load("model_prediksi_do.pkl")
    if isinstance(model, list):
        model = model[0]  # Ambil model dari list jika perlu
    return model

data = load_data()
model = load_model()

# -------------------------------
# SIDEBAR INPUT
# -------------------------------
st.sidebar.title("Input Data Mahasiswa")
ipk_sem1 = st.sidebar.slider("IPK Semester 1", 0.0, 4.0, 3.0, 0.01)
ipk_sem2 = st.sidebar.slider("IPK Semester 2", 0.0, 4.0, 3.0, 0.01)
ipk_sem3 = st.sidebar.slider("IPK Semester 3", 0.0, 4.0, 3.0, 0.01)
ipk_sem4 = st.sidebar.slider("IPK Semester 4", 0.0, 4.0, 3.0, 0.01)

# Pekerjaan dengan urutan sesuai model
pekerjaan_options = ['Pekerjaan Penuh Waktu', 'Paruh Waktu', 'Tidak Bekerja']
pekerjaan = st.sidebar.selectbox("Status Pekerjaan", pekerjaan_options)
pekerjaan_encoded = pekerjaan_options.index(pekerjaan)

kehadiran = st.sidebar.slider("Kehadiran_Rata", 0, 100, 80)
remedial = st.sidebar.number_input("Remedial_Total", 0, 10, 1)
jam_kerja = st.sidebar.slider("Jam_Kerja_Mingguan", 0, 60, 10)
aktivitas_online = st.sidebar.slider("Aktivitas_Online", 0, 50, 20)
pendapatan_ortu = st.sidebar.slider("Pendapatan Orang Tua (juta)", 0, 50, 5)
tanggungan_keluarga = st.sidebar.number_input("Jumlah Tanggungan Keluarga", 0, 10, 2)

# Susun input sesuai dengan urutan fitur saat training
input_data = pd.DataFrame([[
    ipk_sem1, ipk_sem2, ipk_sem3, ipk_sem4,
    pekerjaan_encoded, kehadiran, remedial,
    jam_kerja, aktivitas_online, pendapatan_ortu, tanggungan_keluarga
]], columns=[
    'IPK_Sem1', 'IPK_Sem2', 'IPK_Sem3', 'IPK_Sem4',
    'Pekerjaan', 'Kehadiran_Rata', 'Remedial_Total',
    'Jam_Kerja_Mingguan', 'Aktivitas_Online',
    'Pendapatan_OrangTua', 'Tanggungan_Keluarga'
])

# -------------------------------
# HEADER & TABS
# -------------------------------
st.title("🎓 Dashboard Prediksi Risiko Drop Out (DO) Mahasiswa")
st.markdown("Dashboard ini memprediksi kemungkinan risiko mahasiswa mengalami DO berdasarkan variabel akademik dan aktivitas mingguan.")

tab1, tab2, tab3 = st.tabs(["📊 Eksplorasi Data", "🎯 Prediksi Mahasiswa", "📈 Visualisasi Fitur"])

# -------------------------------
# TAB 1: EDA
# -------------------------------
with tab1:
    st.subheader("Preview Dataset")
    st.dataframe(data.head())

    st.subheader("Statistik Deskriptif")
    st.dataframe(data.describe())

    st.subheader("Distribusi Risiko DO")
    st.bar_chart(data['Risiko_DO'].value_counts())

# -------------------------------
# TAB 2: PREDIKSI
# -------------------------------
with tab2:
    st.subheader("Prediksi Mahasiswa Baru")
    st.write("Masukkan data mahasiswa pada sidebar dan klik tombol prediksi.")
    
    if st.button("Prediksi Risiko DO"):
        try:
            # Gunakan input_data.values agar tidak error nama kolom
            pred = model.predict(input_data.values)[0]
            prob = model.predict_proba(input_data.values)[0][1]
            
            if pred == 1:
                st.error(f"⚠️ Mahasiswa berisiko DO (Probabilitas: {prob:.2f})")
            else:
                st.success(f"✅ Mahasiswa tidak berisiko DO (Probabilitas: {prob:.2f})")
            
            st.write("Data Mahasiswa:")
            st.dataframe(input_data)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")


# -------------------------------
# TAB 3: VISUALISASI
# -------------------------------
with tab3:
    st.subheader("Visualisasi Boxplot Fitur terhadap Risiko DO")
    fitur_plot = ['Kehadiran_Rata', 'Remedial_Total', 'Jam_Kerja_Mingguan', 'Aktivitas_Online']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, col in enumerate(fitur_plot):
        sns.boxplot(data=data, x='Risiko_DO', y=col, ax=axes[i], palette='Set2')
        axes[i].set_title(f'{col} vs Risiko DO')

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Feature Importance")
    try:
        importances = model.feature_importances_
        feat_names = input_data.columns
        imp_df = pd.Series(importances, index=feat_names).sort_values(ascending=False)
        st.bar_chart(imp_df)
    except:
        st.info("Model tidak mendukung feature_importances_.")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""---""")
st.caption("Dibuat Kelompok 13 Data Mining - UAS DATA MINING")
