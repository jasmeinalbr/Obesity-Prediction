# Laporan Proyek Machine Learning - Jasmein Al-baar Putri Rus'an

## **Domain Proyek**

Obesitas merupakan salah satu masalah kesehatan global yang terus meningkat dan menimbulkan dampak serius pada kualitas hidup masyarakat. Berdasarkan data dari World Health Organization [(WHO, 2023](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight), lebih dari 1 miliar orang di seluruh dunia mengalami kelebihan berat badan, termasuk 650 juta orang dewasa yang tergolong obesitas. Prevalensinya telah meningkat tiga kali lipat sejak 1975, dan diperkirakan akan menyebabkan lebih dari 167 juta orang mengalami dampak kesehatan yang serius pada tahun 2025. Kondisi ini meningkatkan risiko berbagai penyakit kronis seperti diabetes tipe 2, penyakit jantung, kanker, hingga gangguan metabolisme lainnya.

Obesitas bukan hanya disebabkan oleh kelebihan konsumsi kalori, tetapi juga merupakan kondisi kompleks yang melibatkan interaksi antara faktor biologis, perilaku, sosial, dan lingkungan. Dalam kajian sistematik oleh [Tandiono & Sanjaya (2023)](https://doi.org/10.33379/gtech.v8i1.3604), ditemukan bahwa penyebab obesitas mencakup faktor genetik, aktivitas fisik rendah, kebiasaan makan tidak sehat, kurang tidur, stres, hingga pengaruh lingkungan seperti akses terhadap makanan cepat saji dan gaya hidup sedentari. Studi tersebut juga menunjukkan bahwa machine learning telah menjadi pendekatan yang menjanjikan dalam mengklasifikasikan tingkat obesitas dan mengidentifikasi pola risiko berdasarkan data individu, serta menjadi alat bantu yang efektif dalam pengambilan keputusan preventif di bidang kesehatan.

Sejalan dengan tantangan tersebut, proyek ini bertujuan untuk membangun model machine learning yang mampu memprediksi kategori obesitas seseorang berdasarkan kombinasi fitur demografis, antropometrik, dan kebiasaan gaya hidup. Melalui pemanfaatan dataset dari Kaggle, proyek ini diharapkan mampu memberikan kontribusi dalam mendeteksi obesitas lebih dini, serta menyediakan wawasan berbasis data dalam memahami faktor risiko obesitas.

## Business Understanding

### Problem Statements
- Apa saja faktor yang berpengaruh terhadap tingkat obesitas seseorang?
- Dapatkah model machine learning secara akurat memprediksi kategori obesitas berdasarkan data karakteristik individu dan gaya hidup?

### Goals
- Mengidentifikasi faktor-faktor utama yang berkontribusi terhadap risiko obesitas melalui analisis fitur dan hasil model.
- Mengembangkan model machine learning untuk memprediksi tingkat obesitas seseorang berdasarkan fitur seperti usia, jenis kelamin, tinggi badan, berat badan, aktivitas fisik, pola makan, dan gaya hidup lainnya.

    ### Solution statements
    - Solusi 1 (Baseline): Membangun model klasifikasi menggunakan algoritma Logistic Regression, yang sederhana dan interpretatif.
    - Solusi 2: Membangun model menggunakan algoritma Random Forest Classifier, yang mampu menangani data campuran (numerik & kategorikal) dan menghasilkan performa yang stabil.
    - Improvement: Melakukan hyperparameter tuning pada model terbaik menggunakan GridSearchCV.
    - Metrik Evaluasi: Menggunakan Akurasi, Precision, Recall, dan F1-score sebagai metrik utama dalam menilai kinerja model klasifikasi.

## Data Understanding

Pada proyek ini, saya menggunakan Obesity Prediction Dataset yang diambil dari Kaggle. Dataset ini dirancang untuk membantu menganalisis faktor-faktor yang berkontribusi terhadap obesitas dengan mengumpulkan data tentang usia, jenis kelamin, tinggi badan, berat badan, indeks massa tubuh (BMI), tingkat aktivitas fisik, dan kategori obesitas individu. Data ini sangat berguna untuk memprediksi prevalensi obesitas dan memahami hubungan antara gaya hidup dan risiko obesitas.

### Variabel-variabel pada dataset adalah sebagai berikut:
- Age: Usia individu (numerik).
- Gender: Jenis kelamin individu (kategorikal: 'Male' atau 'Female').
- Height: Tinggi badan individu (numerik, dalam cm).
- Weight: Berat badan individu (numerik, dalam kg).
- BMI (Body Mass Index): Indeks massa tubuh individu (numerik). Nilai BMI dihitung berdasarkan berat badan dan tinggi badan.
- Physical Activity Level: Tingkat aktivitas fisik individu yang dikategorikan dalam beberapa level: 1, 2, 3, dan 4.
- Obesity Category: Kategori obesitas individu berdasarkan nilai BMI, yang mencakup kategori seperti 'Normal weight', 'Overweight', atau 'Obese'.

Sumber Dataset:
Dataset ini tersedia di Kaggle [Obesity Prediction Dataset](https://www.kaggle.com/datasets/mrsimple07/obesity-prediction/data).

## Exploratory Data Analysis

### Jumlah Data dan Fitur
Dataset ini terdiri dari 1.000 data entri dan 7 kolom (fitur), termasuk label target. Fitur-fitur tersebut mencakup informasi demografis, indikator fisik, dan gaya hidup yang berkaitan dengan obesitas. Semua fitur memiliki nilai yang valid dan tidak terdapat data kosong (missing values).

### Tipe Data

Hasil pemeriksaan tipe data menunjukkan bahwa:

- Fitur seperti Age, Height, Weight, BMI, dan Physical Activity Level adalah numerik
- Fitur seperti Gender dan Obesity Category adalah kategorikal

### Tabel Statistik Deskriptif

| Fitur                  | Count  | Mean    | Std Dev | Min   | 25%    | 50%    | 75%    | Max   |
|------------------------|--------|---------|---------|--------|--------|--------|--------|--------|
| Age                   | 1000   | 49.857  | 18.114  | 18.00  | 35.00  | 50.00  | 66.00  | 79.00 |
| Height (cm)           | 1000   | 170.05  | 10.31   | 136.12 | 163.51 | 169.80 | 177.35 | 201.42|
| Weight (kg)           | 1000   | 71.21   | 15.51   | 26.07  | 61.13  | 71.93  | 81.13  | 118.91|
| BMI                   | 1000   | 24.89   | 6.19    | 8.47   | 20.92  | 24.70  | 28.73  | 50.79 |
| PhysicalActivityLevel | 1000   | 2.534   | 1.116   | 1.00   | 2.00   | 3.00   | 4.00   | 4.00  |

Statistik deskriptif untuk fitur numerik menunjukkan rentang nilai yang wajar:
- Age memiliki nilai antara remaja hingga dewasa
- BMI berkisar dari nilai normal hingga obesitas berat
- Weight dan Height juga memiliki variasi yang mencerminkan populasi umum

### Distribusi Data

#### Distribusi Gender
  
![alt text](<assets/distribusi age.png>)

Distribusi gender menunjukkan bahwa data relatif seimbang antara laki-laki dan perempuan.

#### Distribusi Kategori Obesitas
  
![alt text](<assets/distribusi obesity category.png>)

Label target ObesityCategory memiliki 4 kelas utama: Underweight, Normal weight, Overweight, dan Obese. Kelas terbanyak adalah Normal weight, diikuti oleh Overweight, sedangkan Underweight adalah yang paling sedikit. Distribusi ini penting untuk diperhatikan dalam pemilihan metrik evaluasi, karena data tidak sepenuhnya seimbang.

#### Distribusi Usia

![alt text](<assets/distribusi age.png>)

Distribusi usia tersebar merata dari usia 18 hingga 80 tahun, dengan jumlah tertinggi berada di usia 70–80 tahun. Ini menunjukkan variasi umur yang baik dalam dataset.

#### Distribusi Tinggi Badan

![alt text](<assets/distribusi height.png>)

Distribusi tinggi badan berbentuk mendekati normal (bell curve), dengan rata-rata sekitar 165–170 cm.

#### Distribusi Berat Badan

![alt text](<assets/distribusi weight.png>)

Distribusi berat badan 

#### Distribusi BMI

![alt text](<assets/distribusi bmi.png>)

BMI memiliki distribusi normal yang sedikit miring ke kanan (positively skewed), dengan sebagian besar nilai berada dalam rentang normal dan overweight.

#### Distribusi Physical Activity Level

![alt text](<assets/distribusi physical activity level.png>)

Fitur PhysicalActivityLevel memiliki 4 nilai kategori (1 hingga 4) yang hampir seimbang jumlahnya, mencerminkan variasi aktivitas fisik dalam populasi.

#### Korelasi antar Fitur Numerik

![alt text](<assets/heatmap numerik.png>)

Visualisasi korelasi menunjukkan bahwa:
- BMI dan Weight memiliki korelasi sangat kuat (0.86)
- Height dan BMI memiliki korelasi negatif yang signifikan (-0.48)
- Korelasi antara fitur numerik lainnya relatif lemah

#### Outliers

Selama eksplorasi data, dilakukan visualisasi distribusi dan boxplot untuk fitur numerik. Dari hasil tersebut, ditemukan adanya nilai-nilai ekstrem (outlier) pada fitur BMI, Height, dan Weight.

Namun, karena outlier adalah bagian dari variasi alami dalam data kesehatan, keputusan apakah akan menghapus atau mempertahankannya dijelaskan secara lebih lengkap di bagian Data Preparation.

## Data Preparation

Pada tahap ini, dilakukan serangkaian langkah untuk menyiapkan data agar dapat digunakan dalam proses pelatihan model machine learning. Teknik-teknik yang digunakan:

### 1. Penanganan Data Kosong (Missing Values) dan data duplikat

Dataset diperiksa menggunakan fungsi df.info() dan df.isnull().sum(). Hasilnya menunjukkan bahwa tidak terdapat nilai kosong pada seluruh fitur, sehingga tidak diperlukan imputasi atau penghapusan data.

### 2. Penanganan terhadap Outlier

Dari visualisasi boxplot, terlihat adanya outlier pada beberapa fitur numerik seperti BMI, tinggi badan (Height), dan berat badan (Weight). Namun, setelah ditinjau lebih lanjut, diputuskan untuk tidak menghapus outlier tersebut, dengan pertimbangan:
- Nilai-nilai ekstrem tersebut masih masuk akal secara fisiologis, terutama dalam konteks data kesehatan.
- Outlier bisa merepresentasikan kasus penting seperti obesitas ekstrem atau underweight.
- Model yang digunakan seperti Random Forest dikenal cukup robust terhadap outlier, sehingga tidak akan terlalu terpengaruh oleh nilai-nilai tersebut.

Keputusan ini diambil agar model tetap bisa belajar dari variasi data yang luas dan mencerminkan kondisi nyata populasi.

### 3. Pengubahan Tipe Data

Fitur PhysicalActivityLevel diubah menjadi tipe data kategori (category), karena datanya merepresentasikan level aktivitas yang diskrit (1 sampai 4). Ini membantu dalam penanganan data kategorikal dan memperjelas tipe fitur yang digunakan.

### 4. Encoding Fitur Kategorikal

Agar fitur kategorikal dapat digunakan oleh model machine learning, dilakukan encoding dengan urutan dan teknik berikut:
- Gender diubah menggunakan One-Hot Encoding, menghasilkan kolom baru (Gender_Male). One-hot digunakan karena tidak ada hubungan ordinal antara kategori gender.
- PhysicalActivityLevel diubah ke bentuk numerik menggunakan Label Encoding, karena level 1–4 memiliki arti urutan tingkat aktivitas.
- ObesityCategory sebagai label (target) juga diencoding menggunakan LabelEncoder dari scikit-learn. Hasil encoding sebagai berikut:

```bash
{'Normal weight': 0, 'Obese': 1, 'Overweight': 2, 'Underweight': 3}
```

### 5. Standarisasi Fitur Numerik

Fitur numerik BMI, Weight, Height, dan Age memiliki skala yang berbeda-beda. Oleh karena itu, dilakukan proses standarisasi menggunakan StandardScaler agar seluruh fitur berada pada skala yang sama (rata-rata = 0 dan standar deviasi = 1). Standarisasi ini penting terutama untuk model seperti Logistic Regression, yang sensitif terhadap skala fitur.

### 6. Tampilan Akhir Dataset Setelah Preprocessing

Setelah seluruh proses data preparation dilakukan, termasuk encoding dan standarisasi, berikut adalah **contoh tampilan dataset hasil preprocessing**:

| Age     | Height  | Weight  | BMI     | PhysicalActivityLevel | ObesityCategory | Gender_Male |
|---------|---------|---------|---------|------------------------|------------------|--------------|
| 0.339   | 0.342   | 0.050   | -0.161  | 3                      | 0                | True         |
| 1.057   | -0.575  | 1.210   | 1.374   | 1                      | 1                | True         |
| -0.213  | -0.192  | 0.111   | 0.150   | 3                      | 2                | False        |
| -0.986  | -0.155  | 0.883   | 0.812   | 2                      | 2                | True         |
| 0.560   | 1.312   | -0.140  | -0.711  | 2                      | 0                | True         |

- Kolom `Age`, `Height`, `Weight`, dan `BMI` telah **distandarisasi**.
- `Gender` telah diubah menjadi kolom boolean `Gender_Male` melalui **one-hot encoding**.
- `PhysicalActivityLevel` digunakan sebagai kategori numerik.
- `ObesityCategory` telah diencoding sebagai label numerik (target klasifikasi).

Dengan bentuk akhir seperti ini, dataset telah siap digunakan untuk proses pelatihan model machine learning.

### 7. Split Data (Train-Test Split)

Setelah semua fitur siap, dataset dibagi menjadi dua bagian:
- Data Latih (Training set): 80%
- Data Uji (Testing set): 20%
Pembagian dilakukan dengan menggunakan fungsi train_test_split dari scikit-learn, dan parameter random_state=42 digunakan untuk menjaga hasil yang konsisten saat re-running model.

Setelah dilakukan proses split, didapatkan:
- Dimensi data training: (800, 6) untuk fitur (X_train) dan (800,) untuk label (y_train)
- Dimensi data testing: (200, 6) untuk fitur (X_test) dan (200,) untuk label (y_test)
Ini menunjukkan bahwa data terbagi secara proporsional 80:20, dengan total 1.000 sampel.

Dengan seluruh proses di atas, data telah siap sepenuhnya untuk digunakan dalam tahap pemodelan. Semua tahapan data preparation dilakukan secara sistematis dan logis untuk memastikan kualitas input ke model.

## **Modeling**

Pada tahap ini dilakukan proses pelatihan model machine learning untuk menyelesaikan permasalahan klasifikasi kategori obesitas. Dua algoritma dipilih sesuai dengan solution statement, yaitu:

### 1. Logistic Regression (Baseline)

Logistic Regression digunakan sebagai baseline karena:

- Sederhana dan cepat dilatih
- Memiliki interpretabilitas tinggi (mudah dijelaskan dan dimengerti)
- Cocok untuk klasifikasi multi-kelas yang tidak kompleks

Namun, Logistic Regression memiliki kekurangan dalam menangani **data yang non-linear** atau jika terdapat **hubungan kompleks antar fitur**, sehingga performanya bisa kalah dibanding model yang lebih fleksibel.

### 2. Random Forest Classifier

Random Forest adalah algoritma ensemble yang membentuk banyak decision tree dan menggabungkannya untuk menghasilkan prediksi akhir. Model ini dipilih karena:

- Mampu menangani fitur numerik dan kategorikal
- Robust terhadap overfitting
- Memberikan hasil yang stabil pada data yang kompleks
- Dapat menghitung **feature importance** secara otomatis

Kekurangannya adalah:
- Membutuhkan waktu dan memori lebih besar dibanding model sederhana
- Kurang interpretatif dibanding Logistic Regression

### 3. Evaluasi Awal

Evaluasi awal dilakukan terhadap kedua model menggunakan metrik:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.965    | 0.9658    | 0.965  | 0.9651   |
| Random Forest       | 0.995    | 0.9951    | 0.995  | 0.9950   |

 Dari hasil di atas, **Random Forest jauh lebih unggul** dibanding Logistic Regression dalam semua metrik. Ini menunjukkan bahwa model ini mampu menangkap kompleksitas data dengan lebih baik.

### 4. Hyperparameter Tuning (Improvement)

Untuk memaksimalkan performa Random Forest, dilakukan proses **hyperparameter tuning** menggunakan **GridSearchCV**. Tujuan utama dari tuning ini adalah:

- Mencegah **overfitting**
- Meningkatkan **generalisasi model** terhadap data baru
- Menemukan kombinasi parameter terbaik berdasarkan evaluasi cross-validation

Parameter grid yang digunakan:
```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
```

Setelah proses 3-fold cross-validation, diperoleh kombinasi parameter terbaik:
```python
Best Parameters: {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 100}
```

Model dengan parameter ini digunakan sebagai **model final**.

## **Evaluation**

### Metrik Evaluasi yang Digunakan

Beberapa metrik digunakan untuk mengevaluasi performa akhir model:

1. **Accuracy**  
   Proporsi prediksi yang benar dari total prediksi.  
   ```
   Accuracy = (Jumlah prediksi benar) / (Total jumlah data)
   ```

2. **Precision (Weighted)**  
   Proporsi prediksi positif yang benar, dengan penyesuaian jumlah data tiap kelas.  
   ```
   Precision = TP / (TP + FP)
   ```

3. **Recall (Weighted)**  
   Kemampuan model menemukan semua instance dari tiap kelas.  
   ```
   Recall = TP / (TP + FN)
   ```

4. **F1 Score (Weighted)**  
   Rata-rata harmonis dari precision dan recall.  
   ```
   F1 = 2 * (Precision * Recall) / (Precision + Recall)
   ```

---

### Hasil Evaluasi Akhir (Random Forest Tuned)

Setelah dilakukan pelatihan dan tuning, model Random Forest terbaik memberikan hasil evaluasi sebagai berikut:

| Metrik      | Nilai   |
|-------------|---------|
| Accuracy    | 0.995   |
| Precision   | 0.9951  |
| Recall      | 0.9950  |
| F1 Score    | 0.9950  |

Model juga dievaluasi menggunakan **confusion matrix**, yang menunjukkan bahwa:

- Hampir seluruh kelas diklasifikasikan dengan sangat akurat
- Tidak ada kelas yang tertinggal secara signifikan
- Kelas **Obese** memiliki recall sedikit lebih rendah (~0.97), namun precision tetap sempurna (1.00)

📊 *Visualisasi confusion matrix* ditampilkan di bagian notebook sebagai pendukung interpretasi per kelas.


### Kesimpulan Evaluasi

Model **Random Forest dengan hyperparameter tuning** dipilih sebagai model akhir karena:

- Mencapai performa tertinggi secara konsisten
- Menghasilkan klasifikasi yang akurat di semua kelas
- Memiliki keseimbangan precision dan recall yang sangat baik

Model ini dapat diandalkan untuk digunakan dalam sistem prediksi obesitas berbasis machine learning dengan tingkat kepercayaan tinggi.

## **Inference**

Setelah model terbaik (Random Forest hasil hyperparameter tuning) dipilih, dilakukan proses **inferensi** terhadap data baru untuk melihat bagaimana model digunakan dalam praktik nyata.

### Data Uji Baru
Data uji baru yang digunakan memiliki fitur sebagai berikut:

| Fitur                  | Nilai   |
|------------------------|---------|
| Age                   | 30      |
| Height (cm)           | 170     |
| Weight (kg)           | 70      |
| BMI                   | 24.2    |
| PhysicalActivityLevel | 3       |
| Gender_Male           | 1       |

Sebelum dilakukan prediksi, fitur numerik (`Age`, `Height`, `Weight`, `BMI`) distandarisasi menggunakan scaler yang sama dengan yang digunakan saat training.

### Hasil Prediksi

- **Prediksi Kategori Obesitas**:  
  ```
  Normal weight
  ```

- **Probabilitas Prediksi per Kategori**:

| Kategori        | Probabilitas |
|------------------|--------------|
| Normal weight   | 0.9659       |
| Overweight      | 0.0241       |
| Underweight     | 0.0100       |
| Obese           | 0.0000       |

Hasil prediksi menunjukkan bahwa individu dengan karakteristik tersebut diperkirakan berada pada kategori **Normal weight**, dengan tingkat keyakinan model yang sangat tinggi (**96.59%**).

---

### Catatan
Proses inference ini menunjukkan bagaimana model dapat digunakan untuk:
- Memprediksi kategori obesitas dari data individu baru
- Memberikan probabilitas keyakinan model terhadap setiap kelas
- Menjadi dasar pengambilan keputusan preventif berbasis data di bidang kesehatan
