from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE = Path(__file__).resolve().parent
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = error = None
    temperature = ''
    model_path = BASE / 'model.joblib'
    metrics_path = BASE / 'metrics.json'
    metrics = json.loads(metrics_path.read_text(encoding='utf-8')) if metrics_path.exists() else {}
    model = joblib.load(model_path) if model_path.exists() else None
    if request.method == 'POST':
        temperature = request.form.get('temperature', '').strip()
        try:
            value = float(temperature)
            if not 0 <= value <= 50:
                raise ValueError('Suhu harus berada di antara 0 sampai 50 °C.')
            if model is None:
                error = 'Model belum tersedia. Jalankan train_model.py terlebih dahulu.'
            else:
                prediction = max(0.0, float(model.predict(pd.DataFrame({'T1': [value]}))[0]))
        except ValueError as exc:
            error = str(exc) or 'Masukkan suhu dalam bentuk angka.'
    return render_template('index.html', prediction=prediction, error=error, temperature=temperature, metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True)
