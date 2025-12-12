# Source Code

Folder ini berisi modul Python yang dapat digunakan kembali (reusable) dan dipanggil oleh notebook atau script training.

## Daftar File

- **`data_loader.py`**: Fungsi untuk memuat data raw, melakukan preprocessing fitur (RSI, MACD), scaling, dan splitting data (Train/Val/Test).
- **`model.py`**: Definisi arsitektur Deep Learning. Menggunakan **Bi-LSTM** dengan **Custom Attention Layer** untuk menangkap pola urutan waktu (time-series).
- **`train.py`**: Script utama untuk melatih model. Mengatur training loop, optimizer, loss function, dan callback (EarlyStopping).
- **`utils.py`**: Fungsi bantuan untuk plotting grafik loss/akurasi dan utilitas umum lainnya.
