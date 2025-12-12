# Outputs

Folder ini menyimpan semua artefak hasil dari proses training model.

## Struktur
- **models/**: Menyimpan file model yang sudah dilatih (`.keras`) dan objek scaler (`scaler_data.pkl`) untuk kebutuhan deployment.
- **logs/**: Menyimpan file CSV log training (loss & metric per epoch) untuk analisis performa.
- **plots/**: Menyimpan gambar hasil visualisasi, seperti grafik Loss Curve, Confusion Matrix, dan perbandingan prediksi vs aktual.
