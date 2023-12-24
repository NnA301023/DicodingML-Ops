# Submission 1: ML Pipeline - Intent Classification on ATIS Airline

Username Dicoding: nna_alif

![](https://miro.medium.com/v2/resize:fit:1400/1*Xe8qYW2BdcWc1U5PRCgoXw.png)


| Informasi Umum | Deskripsi | 
| --- | --- |
| Dataset | [ATIS Airline Travel Information System](https://www.kaggle.com/datasets/hassanamin/atis-airlinetravelinformationsystem/data?select=atis_intents.csv) |
| Masalah | Memahami dengan akurat intensi pengguna menjadi hal yang krusial untuk meningkatkan efektivitas interaksi chatbot, sehingga bisa memberikan response yang lebih tersegmentasi dan meningkatkan convention rate melalui percakapan teks. |
| Solusi Machine Learning | Dikarenakan banyaknya intensi interaksi antara pengguna dengan chatbot, kami mendeterminasi intensi hanya seputar `penerbangan` dengan konsep label intensi terbagi menjadi `Terkait Penerbangan` (1) dan `Diluar Penerbangan` (0), sehingga mampu memberikan funnel lanjutan terhadap response yang diberikan dan meningkatkan potensi convension rate di konteks `penerbangan` |
| Metode Pengolahan | Dengan dataset yang terdiri dari 2 kolom `message` dan `is_flight_intent` kami membagi dataset menjadi partisi training (80%) dan evaluation (20%), kemudian kami meninjau lebih lanjut terkait aspek statistik di dalam dataset termasuk tipe data dan anomali pada dataset, kemudian feature `message` kami transformasi ke dalam bentuk vektor menggunakan fungsi  `TextVectorization()` dan untuk feature `label` hanya kami casting / pengubahan tipe data menggunakan fungsi `tf.cast()` menjadi tipe data int64|
| Arsitektur Model | Menggunakan `LSTM` sebagai hiddenn layer utama, juga termasuk lapisan `Dense` dan `Dropout` untuk meningkatkan transfer bobot untuk menentukan apakah pesan adalah `Terkait Penerbangan` (1) dan `Diluar Penerbangan` (0) menggunakan aktivasi `Sigmoid` dan mengevaluasi model menggunakan `BinaryCrossEntropy` |
| Metrik Evaluasi | Metrik evaluasi yang digunakan yaitu ExampleCount, AUC, FalsePositives, TruePositives, FalseNegatives, TrueNegatives, dan BinaryAccuracy |
| Performa Model | Menurut metrik `Binary Accuracy`, untuk mengevaluasi training set (99,34%) dan evaluation set (98,91%) untuk menentukan apakah `message` terdapat dalam intent `Terkait Penerbangan` (1) atau `Diluar Penerbangan` (0), sudah menunjukkan bahwa arsitektur LSTM memberikan performa yang baik. |