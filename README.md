# Proyek Analisis Overfitting & Underfitting: Implementasi ML Workflow

Repositori ini memuat implementasi praktis dari *Machine Learning Workflow* pada tahap **Model Evaluation (Evaluasi Model)**. Proyek ini berfokus pada simulasi, identifikasi, dan strategi penanganan dua masalah utama pada performa model: **Overfitting** (model terlalu kompleks) dan **Underfitting** (model terlalu sederhana).

---

## Kasus 1: Mengatasi Underfitting (Studi Kasus Klasifikasi)

Kasus ini disimulasikan menggunakan *Breast Cancer Dataset* untuk mendeteksi kanker payudara (klasifikasi biner).

### Identifikasi Underfitting
* **Kondisi Awal:** Model dilatih menggunakan algoritma `DecisionTreeClassifier` dengan membatasi kedalaman pohon secara ekstrem (`max_depth=1`).
* **Gejala Performa:** Akurasi pada data latih (*Training Accuracy*) dan data uji (*Test Accuracy*) sama-sama rendah. Model gagal menangkap pola mendasar yang ada pada data karena terlalu sederhana (*High Bias*).

### Strategi Penanganan Underfitting
Proyek ini mendemonstrasikan beberapa cara efektif untuk keluar dari kondisi underfitting:
1.  **Meningkatkan Kompleksitas Model:** Menambah parameter kedalaman maksimum pohon (`max_depth=10`) agar algoritma diberikan ruang untuk mempelajari aturan keputusan yang lebih mendalam dan spesifik.
2.  **Feature Scaling (Normalisasi data):** Mentransformasikan fitur menggunakan `StandardScaler` agar variasi rentang nilai antar-fitur tidak membingungkan algoritma saat memetakan batas keputusan.
3.  **Menambah Ukuran Data Latih:** Mengatur ulang proporsi *data splitting* (`test_size=0.2`) guna memberikan lebih banyak suplai variasi sampel data pada tahap *training set* (`X_train_more_data`). Hasilnya, akurasi data latih meningkat drastis hingga mencapai 100% dan akurasi data uji naik signifikan ke angka ~94%.

---

## Kasus 2: Mengatasi Overfitting (Studi Kasus Regresi)

Kasus ini disimulasikan menggunakan *California Housing Dataset* untuk memprediksi harga median rumah (regresi nilai kontinu).

### Identifikasi Overfitting
* **Kondisi Awal:** Model regresi dilatih menggunakan `DecisionTreeRegressor` dengan membiarkan pohon tumbuh secara maksimal tanpa batasan kedalaman (*unconstrained tree*).
* **Gejala Performa:** Nilai *Training Error* (MSE) mendekati angka 0 (sempurna), namun nilai *Test Error* (MSE) membengkak sangat besar. Model menghafal seluruh data latih beserta *noise*-nya secara berlebihan, sehingga gagal melakukan generalisasi pada data baru (*High Variance*).

### Strategi Penanganan Overfitting
Proyek ini menerapkan beberapa teknik regularisasi dan arsitektur untuk menekan efek overfitting:
1.  **Pemangkasan Pohon (Pruning / Constraint):** Membatasi kedalaman maksimum pohon keputusan (misal, `max_depth=5`) untuk menghentikan model menghafal detail sampel secara spesifik.
2.  **Metode Ensemble (Random Forest Regressor):** Mengganti model tunggal dengan `RandomForestRegressor` yang menggabungkan hasil prediksi dari 100 pohon keputusan acak (`n_estimators=100`). Pendekatan *bagging* ini terbukti ampuh menurunkan nilai *Test MSE* secara drastis karena meratakan varians model.
3.  **Cross-Validation:** Menerapkan evaluasi berbasis `cross_val_score` untuk memastikan kestabilan performa model di berbagai sub-sampel data yang berbeda sebelum masuk fase produksi.

---

## Library Python Utama yang Digunakan
* `sklearn.datasets` (`load_breast_cancer`, `fetch_california_housing`) - Memuat dataset standar untuk eksperimen evaluasi performa.
* `sklearn.tree` (`DecisionTreeClassifier`, `DecisionTreeRegressor`) - Digunakan sebagai model dasar untuk mensimulasikan karakteristik bias dan varians.
* `sklearn.ensemble` (`RandomForestRegressor`) - Algoritma berbasis *ensemble* untuk mereduksi efek *overfitting*.
* `sklearn.metrics` (`accuracy_score`, `mean_squared_error`) - Metrik utama untuk memantau celah performa (*generalization gap*) antara data latih dan data uji.
