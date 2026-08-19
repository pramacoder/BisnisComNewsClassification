# Bisnis.com News Classification

Proyek ini bertujuan untuk mengklasifikasikan artikel berita dari **Bisnis.com** ke dalam 11 kategori (kelas) berbeda menggunakan berbagai pendekatan Machine Learning dan Deep Learning, termasuk *baseline* tradisional dan model berbasis *Neural Network*.

## 📊 Dataset
Dataset terdiri dari artikel berita Bisnis.com (tahun 2020 - 2022) yang dibagi menjadi:
- `train.csv` (Data Latih)
- `val.csv` (Data Validasi)
- `test.csv` (Data Uji)

Terdapat 11 kategori/kelas berita, antara lain: APBN, Agribisnis, Ekonomi, Ekonomi Global, Energi & Tambang, Infrastruktur, Jasa & Niaga, Manufaktur, Pajak, Properti, dan Transportasi & Logistik.

## ⚙️ Preprocessing
Tahapan pra-pemrosesan teks meliputi:
- Case folding (lowercasing)
- Pembersihan tanda baca, angka, dan karakter khusus
- Stopwords removal (menghilangkan kata hubung yang tidak bermakna)
- Stemming (menggunakan Sastrawi)

## 🤖 Model yang Dievaluasi
Terdapat 3 kelompok pendekatan model yang dieksplorasi:

### 1. Baseline Machine Learning (TF-IDF)
Representasi fitur menggunakan TF-IDF (Term Frequency-Inverse Document Frequency) dengan klasifikasi tradisional:
- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **Multinomial Naive Bayes**

### 2. Deep Learning (Word Embeddings)
Model yang dilatih dari awal menggunakan representasi *word embedding*:
- **CNN (Convolutional Neural Network)**
- **BiLSTM (Bidirectional Long Short-Term Memory)**
- **CNN-BiLSTM** (Kombinasi CNN untuk ekstraksi fitur spasial/lokal dan BiLSTM untuk fitur sekuensial)

### 3. Transformers (Pre-trained)
- **IndoBERT** (`indobenchmark/indobert-base-p1`) (Dioptimasi di Google Colab)

## 📈 Hasil Evaluasi (Test Set)

Berdasarkan evaluasi terhadap data uji (test set), berikut adalah ringkasan kinerja masing-masing model:

| Tipe | Model | Accuracy | F1-Score (Weighted) |
|---|---|:---:|:---:|
| **Transformer** | 🥇 **IndoBERT** (Colab) | **0.9021** | **0.9025** |
| **Baseline** | 🥈 **SVM** | 0.9012 | 0.9004 |
| **Baseline** | Logistic Regression | 0.8926 | 0.8914 |
| **Baseline** | Naive Bayes | 0.8416 | 0.8297 |
| **Deep Learning** | CNN | 0.8777 | 0.8725 |
| **Deep Learning** | **BiLSTM** | **0.8308** | **0.8180** |
| **Deep Learning** | CNN-BiLSTM | 0.8247 | 0.8151 |

### 🔍 Analisis Performa BiLSTM
- **BiLSTM** mendapatkan akurasi **83.08%** dan F1-Score **81.80%**.
- Meskipun BiLSTM sangat baik dalam menangkap dependensi jangka panjang pada teks (*sequential context*), performanya dalam dataset ini **kalah dibandingkan CNN (87.77%)** dan **SVM (90.12%)**.
- **Alasan yang mungkin:** Klasifikasi topik berita biasanya lebih bergantung pada *keyword* spesifik (yang dengan sangat baik ditangkap oleh TF-IDF + SVM atau *n-gram detectors* pada CNN) dibandingkan struktur urutan kata yang kompleks. 
- Kombinasi **CNN-BiLSTM (82.47%)** justru mengalami sedikit penurunan dibanding BiLSTM murni, mengindikasikan kemungkinan model *overfitting* atau arsitektur yang terlalu kompleks untuk ukuran dataset.

## 🚀 Kesimpulan & Next Steps
1. 🥇 **IndoBERT (Fine-Tuning)** terbukti menjadi model dengan performa terbaik pada proyek ini, mencapai akurasi **90.21%** dan F1-Score **90.25%**. Kemampuannya dalam memahami konteks kalimat secara mendalam (*attention mechanism*) berhasil mengalahkan pendekatan *baseline*.
2. 🥈 **SVM dengan TF-IDF** (Akurasi 90.12%) menjadi alternatif kedua yang sangat kuat dan jauh lebih efisien komputasinya. Hal ini membuktikan bahwa klasifikasi topik berita seringkali bisa diselesaikan dengan baik dengan mendeteksi kemunculan *keyword* spesifik (yang ditangkap kuat oleh TF-IDF).
3. **Model Deep Learning murni (BiLSTM/CNN)** mendapat skor di kisaran 82-87%, yang mengindikasikan mungkin perlunya tuning yang lebih dalam, penambahan *pre-trained word embeddings* (seperti FastText/Word2Vec), atau lebih banyak data latih untuk mengungguli performa TF-IDF dan Transformers.

## 📂 Struktur Proyek
- `/data`: Berisi file CSV (train, val, test)
- `/notebooks`: Kumpulan Jupyter Notebook untuk EDA, Training Baseline, DL, dan IndoBERT.
- `/src`: Skrip Python modular untuk preprocessing, training, dan evaluasi.
- `/outputs`: Hasil prediksi, confusion matrix, laporan klasifikasi, grafik *training history*, dan log metrik model.
- `/models`: Folder untuk menyimpan model yang telah dilatih (mis. *label encoder*, *checkpoints*).
