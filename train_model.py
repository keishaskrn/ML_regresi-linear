from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
DATA = BASE / 'dataset' / 'energydata_complete.csv'
if not DATA.exists():
    raise FileNotFoundError(f'Dataset belum ditemukan: {DATA}\nUnduh dan ekstrak CSV dari Kaggle lalu simpan di folder dataset.')
df = pd.read_csv(DATA)
needed = ['T1', 'Appliances']
if any(c not in df.columns for c in needed):
    raise ValueError(f'Dataset harus memiliki kolom {needed}')
data = df[needed].apply(pd.to_numeric, errors='coerce').dropna()
X, y = data[['T1']], data['Appliances']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
metrics = {'rows_used': int(len(data)), 'train_rows': int(len(X_train)), 'test_rows': int(len(X_test)),
           'feature': 'T1', 'target': 'Appliances', 'intercept': float(model.intercept_),
           'coefficient': float(model.coef_[0]), 'mae': float(mean_absolute_error(y_test, pred)),
           'rmse': float(np.sqrt(mean_squared_error(y_test, pred))), 'r2': float(r2_score(y_test, pred))}
joblib.dump(model, BASE / 'model.joblib')
(BASE / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
plt.figure(figsize=(8,5)); plt.scatter(X_test['T1'], y_test, alpha=.35, label='Data aktual')
order = np.argsort(X_test['T1'].to_numpy()); plt.plot(X_test['T1'].to_numpy()[order], pred[order], linewidth=2, label='Garis regresi')
plt.xlabel('Suhu dapur T1 (°C)'); plt.ylabel('Penggunaan energi Appliances (Wh)'); plt.title('Suhu Dapur dan Penggunaan Energi'); plt.legend(); plt.tight_layout(); plt.savefig(BASE/'grafik_scatter_t1_appliances.png', dpi=150); plt.close()
plt.figure(figsize=(6,6)); plt.scatter(y_test, pred, alpha=.4); lo=float(min(y_test.min(), pred.min())); hi=float(max(y_test.max(), pred.max())); plt.plot([lo,hi],[lo,hi],'--')
plt.xlabel('Nilai aktual (Wh)'); plt.ylabel('Nilai prediksi (Wh)'); plt.title('Nilai Aktual vs Prediksi'); plt.tight_layout(); plt.savefig(BASE/'grafik_aktual_vs_prediksi.png', dpi=150); plt.close()
print('=== HASIL MODEL REGRESI LINEAR ===')
for k, label in [('rows_used','Jumlah data'),('train_rows','Data training'),('test_rows','Data testing'),('feature','Fitur X'),('target','Target y'),('intercept','Intercept'),('coefficient','Koefisien'),('mae','MAE'),('rmse','RMSE'),('r2','R-squared')]:
    v = metrics[k]; print(f'{label:22}: {v:.4f}' if isinstance(v, float) else f'{label:22}: {v}')
print('Model tersimpan: model.joblib'); print('Grafik tersimpan: grafik_scatter_t1_appliances.png dan grafik_aktual_vs_prediksi.png')
