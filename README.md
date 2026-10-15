# TA-02 — Regresi Linear dengan Flask

Aplikasi web untuk memperkirakan penggunaan energi peralatan rumah berdasarkan suhu dapur (`T1`) dari dataset Appliances Energy Prediction.

## Jalankan di VS Code (PowerShell)
1. Unduh dataset dari https://www.kaggle.com/datasets/loveall/appliances-energy-prediction, ekstrak ZIP, dan salin `energydata_complete.csv` ke folder `dataset`.
2. Jalankan:
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py
python app.py
```
3. Buka `http://127.0.0.1:5000`.

## Deployment
- Pastikan CSV berada di `dataset/energydata_complete.csv` pada repository.
- Build command: `pip install -r requirements.txt && python train_model.py`
- Start command: `gunicorn app:app`
- Hubungkan repository GitHub ke Render atau Railway dan uji URL publik.

## Isi laporan
Masukkan screenshot kode dan output pelatihan, grafik, halaman aplikasi dan contoh prediksi. Lengkapi tautan GitHub, hosting, serta video YouTube setelah tersedia. Nilai R² yang rendah harus dilaporkan apa adanya sebagai keterbatasan model satu fitur.
