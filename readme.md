# Cryptocurrency Movement and Volatility Prediction

Proyek ini bertujuan untuk memprediksi pergerakan pasar mata uang kripto (Bitcoin) menggunakan pendekatan **Deep Learning**. Model yang dikembangkan adalah **Bi-LSTM (Bidirectional Long Short-Term Memory)** yang dilengkapi dengan mekanisme **Attention**, yang mampu menangkap pola temporal jangka panjang dari data historis.

Sistem ini memiliki dua luaran utama (Multi-Output):
1.  **Price Direction**: Klasifikasi biner untuk memprediksi apakah harga akan *Naik (UP)* atau *Turun (DOWN)*.
2.  **Volatility Prediction**: Regresi untuk memprediksi tingkat volatilitas pasar.

---

## Link Aplikasi

Akses dashboard prediksi interaktif melalui tautan berikut:

https://waterlemonnn-cryptocurrency-movement-and-volatili-appapp-q0e7p1.streamlit.app/

---

## Detail Teknis

* **Arsitektur Model**: Bi-LSTM (64 units) + Attention Layer + Dense Layers.
* **Fitur Input**: Data OHLCV, Log Return, Log Volatility, Upper/Lower Shadow, RSI, dan MACD Normalized.
* **Optimasi**: Menggunakan optimizer Adam dengan *learning rate* dinamis dan *Early Stopping* untuk mencegah overfitting.
