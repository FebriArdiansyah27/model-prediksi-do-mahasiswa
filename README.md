
# 📘 Laporan Proyek Machine Learning - Prediksi Risiko DO Mahasiswa

## 🧭 Domain Proyek

Pendidikan tinggi merupakan salah satu faktor kunci dalam pembangunan sumber daya manusia. Namun, banyak mahasiswa mengalami kesulitan menyelesaikan studi tepat waktu, bahkan berisiko drop out (DO). Proyek ini bertujuan untuk membangun sistem prediksi dini terhadap risiko DO mahasiswa berdasarkan data akademik, ekonomi, dan aktivitas mereka.

### Mengapa Masalah Ini Penting?

Dengan model prediksi ini, pihak kampus dapat melakukan intervensi lebih awal terhadap mahasiswa yang teridentifikasi berisiko, sehingga mencegah DO dan meningkatkan kualitas pendidikan.


## 💡 Business Understanding

### Problem Statements

- Bagaimana cara memprediksi risiko DO mahasiswa berdasarkan data akademik, kehadiran, dan faktor ekonomi?
- Fitur mana saja yang paling berpengaruh terhadap potensi mahasiswa mengalami DO?

### Goals

- Membangun model klasifikasi untuk memprediksi risiko DO mahasiswa.
- Mengidentifikasi faktor-faktor penting penyebab DO untuk mendukung kebijakan kampus.

### Solution Statements

- Menggunakan algoritma Random Forest sebagai baseline karena kemampuannya dalam menangani data non-linear dan feature selection otomatis.
- Melakukan evaluasi model dengan metrik akurasi dan F1-score serta visualisasi confusion matrix.

## 📊 Data Understanding

Dataset yang digunakan merupakan data sintetis berjumlah 1500 baris dengan 8–9 fitur, yang mencerminkan kondisi akademik, aktivitas, serta kondisi sosial ekonomi mahasiswa.

**Fitur-fitur dalam dataset (versi terbaru):**

- `IPK` : Nilai Rata-rata IP selama kuliah.
- `Kehadiran_Rata` : Persentase kehadiran mahasiswa di kelas.
- `Remedial_Total` : Jumlah mata kuliah yang diulang.
- `Aktivitas_Online` : Rata-rata jam belajar daring per minggu.
- `Pekerjaan` : Apakah mahasiswa bekerja (Ya/Tidak).
- `Jam_Kerja_Mingguan` : Total jam kerja dalam seminggu.
- `Pendapatan_OrangTua` : Pendapatan orang tua dalam juta/bulan.
- `Tanggungan_Keluarga` : Jumlah anggota keluarga yang ditanggung.

**Target klasifikasi:**

- `Label_DO` : 1 = Risiko Drop Out, 0 = Aman.

## 🛠️ Data Preparation

Tahapan persiapan data meliputi:

- Penghapusan IPK per semester: IPK semester 1 hingga 4 diubah menjadi satu nilai rata-rata `IPK`.
- Pembersihan data duplikat: Menghapus 1000 baris duplikat dari total 3000 baris awal.
- Penambahan data unik: Menambahkan 1000 data baru yang tidak duplikat.
- Transformasi kategori: Fitur dengan nilai "Ya/Tidak" diubah menjadi 1/0.
- One-hot encoding: Untuk fitur kategorikal sebelum modeling.
- Normalisasi tidak diperlukan karena model Random Forest tidak sensitif terhadap skala.

## 🤖 Modeling

Model yang digunakan adalah **Random Forest Classifier** dengan parameter default:

```python
RandomForestClassifier(n_estimators=100, random_state=42)
```

Model dilatih pada data training sebesar 80% dari total dataset.

### Kelebihan Random Forest:

- Dapat menangani data numerik dan kategorikal.
- Mengurangi risiko overfitting dibandingkan decision tree tunggal.
- Memiliki kemampuan untuk mengevaluasi pentingnya fitur (*feature importance*).

## ✅ Evaluation

### Metrik evaluasi:

- **Akurasi**: Proporsi prediksi yang benar terhadap seluruh data.
- **Confusion Matrix**: Menunjukkan distribusi prediksi benar dan salah.
- **Classification Report**: Menyediakan precision, recall, dan F1-score.

### Hasil Evaluasi:

- **Akurasi Model**: 92.5%
- **F1-score (DO class)**: 90%
- Confusion Matrix menunjukkan bahwa model cukup seimbang dalam mengenali mahasiswa yang berisiko maupun aman.

Metrik F1-score menjadi penting dalam kasus ini karena menghindari kesalahan memprediksi mahasiswa berisiko sebagai aman (false negative) sangat penting untuk intervensi dini.

## 📎 Lampiran Tambahan

- `model_prediksi_do.pkl`: File model hasil pelatihan.
- `kolom_model.pkl`: File penyimpanan kolom untuk transformasi data baru.
- File `prediksi_interaktif.py`: Berisi antarmuka prediksi berbasis input terminal.

