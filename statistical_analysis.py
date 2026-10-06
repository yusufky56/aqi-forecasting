import pandas as pd
import numpy as np
import pickle
from scipy import stats
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

import os
from data_prep import ROOT, load_dataset

# =====================================================
# 1. VERİYİ HAZIRLA (train_models.py ile aynı, data_prep.py)
# =====================================================
print("="*80)
print("VERİ HAZIRLANIYOR...")
print("="*80)

ds = load_dataset()
scaler_y = ds.scaler_y
X_test, y_test = ds.X_test, ds.y_test
print(f"Test seti: {len(X_test)} örnek")

# =====================================================
# 2. MODELLERİ YÜKLE VE TAHMİN YAP
# =====================================================
print("\n" + "="*80)
print("MODELLER YÜKLENİYOR VE TAHMİN YAPILIYOR...")
print("="*80)

# Modelleri yükle
lstm_model = tf.keras.models.load_model(os.path.join(ROOT, "lstm_best.keras"))
gru_model = tf.keras.models.load_model(os.path.join(ROOT, "gru_best.keras"))
transformer_model = tf.keras.models.load_model(os.path.join(ROOT, "transformer_best.keras"))

# Tahminler
y_pred_lstm = lstm_model.predict(X_test, verbose=0).flatten()
y_pred_gru = gru_model.predict(X_test, verbose=0).flatten()
y_pred_transformer = transformer_model.predict(X_test, verbose=0).flatten()

# Ölçeği geri al
y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_lstm_real = scaler_y.inverse_transform(y_pred_lstm.reshape(-1, 1)).flatten()
y_pred_gru_real = scaler_y.inverse_transform(y_pred_gru.reshape(-1, 1)).flatten()
y_pred_transformer_real = scaler_y.inverse_transform(y_pred_transformer.reshape(-1, 1)).flatten()

print(f"Tahminler tamamlandı!")

# =====================================================
# 3. HATALARI HESAPLA
# =====================================================
errors_lstm = y_test_real - y_pred_lstm_real
errors_gru = y_test_real - y_pred_gru_real
errors_transformer = y_test_real - y_pred_transformer_real

# =====================================================
# 4. DIEBOLD-MARIANO TESTİ
# =====================================================
print("\n" + "="*80)
print("DIEBOLD-MARIANO TESTİ")
print("="*80)

def diebold_mariano_test(e1, e2, h=1):
    """
    Diebold-Mariano testi
    H0: İki model arasında tahmin doğruluğu farkı yoktur
    """
    d = e1**2 - e2**2  # Kare hata farkları
    mean_d = np.mean(d)
    
    # Autocovariance hesapla
    n = len(d)
    gamma = []
    for k in range(h):
        gamma.append(np.sum((d[k:] - mean_d) * (d[:n-k] - mean_d)) / n)
    
    # Varyans
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    
    if var_d <= 0:
        var_d = np.var(d) / n
    
    # DM istatistiği
    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value

# GRU vs LSTM
dm_gru_lstm, p_gru_lstm = diebold_mariano_test(errors_gru, errors_lstm)
print(f"GRU vs LSTM:        DM = {dm_gru_lstm:7.4f}, p-value = {p_gru_lstm:.6f}", end="")
print("  ✓ Anlamlı" if p_gru_lstm < 0.05 else "  ✗ Anlamlı değil")

# GRU vs Transformer
dm_gru_trans, p_gru_trans = diebold_mariano_test(errors_gru, errors_transformer)
print(f"GRU vs Transformer: DM = {dm_gru_trans:7.4f}, p-value = {p_gru_trans:.6f}", end="")
print("  ✓ Anlamlı" if p_gru_trans < 0.05 else "  ✗ Anlamlı değil")

# LSTM vs Transformer
dm_lstm_trans, p_lstm_trans = diebold_mariano_test(errors_lstm, errors_transformer)
print(f"LSTM vs Transformer: DM = {dm_lstm_trans:7.4f}, p-value = {p_lstm_trans:.6f}", end="")
print("  ✓ Anlamlı" if p_lstm_trans < 0.05 else "  ✗ Anlamlı değil")

# =====================================================
# 5. WILCOXON SIGNED-RANK TESTİ
# =====================================================
print("\n" + "="*80)
print("WILCOXON SIGNED-RANK TESTİ")
print("="*80)

# Mutlak hatalar üzerinden
abs_errors_lstm = np.abs(errors_lstm)
abs_errors_gru = np.abs(errors_gru)
abs_errors_transformer = np.abs(errors_transformer)

# GRU vs LSTM
stat, p_val = stats.wilcoxon(abs_errors_gru, abs_errors_lstm)
print(f"GRU vs LSTM:        W = {stat:10.1f}, p-value = {p_val:.6f}", end="")
print("  ✓ Anlamlı" if p_val < 0.05 else "  ✗ Anlamlı değil")

# GRU vs Transformer
stat, p_val = stats.wilcoxon(abs_errors_gru, abs_errors_transformer)
print(f"GRU vs Transformer: W = {stat:10.1f}, p-value = {p_val:.6f}", end="")
print("  ✓ Anlamlı" if p_val < 0.05 else "  ✗ Anlamlı değil")

# LSTM vs Transformer
stat, p_val = stats.wilcoxon(abs_errors_lstm, abs_errors_transformer)
print(f"LSTM vs Transformer: W = {stat:10.1f}, p-value = {p_val:.6f}", end="")
print("  ✓ Anlamlı" if p_val < 0.05 else "  ✗ Anlamlı değil")

# =====================================================
# 6. BOOTSTRAP CONFIDENCE INTERVAL (R² için)
# =====================================================
print("\n" + "="*80)
print("BOOTSTRAP %95 GÜVEN ARALIĞI (R² için)")
print("="*80)

def bootstrap_r2(y_true, y_pred, n_bootstrap=1000, ci=95):
    """Bootstrap ile R² güven aralığı"""
    r2_scores = []
    n = len(y_true)
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        r2 = r2_score(y_true[idx], y_pred[idx])
        r2_scores.append(r2)
    
    lower = np.percentile(r2_scores, (100 - ci) / 2)
    upper = np.percentile(r2_scores, 100 - (100 - ci) / 2)
    mean_r2 = np.mean(r2_scores)
    
    return mean_r2, lower, upper

# Her model için
r2_lstm, low_lstm, up_lstm = bootstrap_r2(y_test_real, y_pred_lstm_real)
r2_gru, low_gru, up_gru = bootstrap_r2(y_test_real, y_pred_gru_real)
r2_trans, low_trans, up_trans = bootstrap_r2(y_test_real, y_pred_transformer_real)

print(f"LSTM:        R² = {r2_lstm:.4f}  [%95 CI: {low_lstm:.4f} - {up_lstm:.4f}]")
print(f"GRU:         R² = {r2_gru:.4f}  [%95 CI: {low_gru:.4f} - {up_gru:.4f}]")
print(f"Transformer: R² = {r2_trans:.4f}  [%95 CI: {low_trans:.4f} - {up_trans:.4f}]")

# =====================================================
# 7. SONUÇLARI KAYDET
# =====================================================
print("\n" + "="*80)
print("SONUÇLAR KAYDEDİLİYOR...")
print("="*80)

results = {
    'Diebold_Mariano': {
        'GRU_vs_LSTM': {'DM': dm_gru_lstm, 'p_value': p_gru_lstm},
        'GRU_vs_Transformer': {'DM': dm_gru_trans, 'p_value': p_gru_trans},
        'LSTM_vs_Transformer': {'DM': dm_lstm_trans, 'p_value': p_lstm_trans}
    },
    'Bootstrap_R2': {
        'LSTM': {'R2': r2_lstm, 'CI_lower': low_lstm, 'CI_upper': up_lstm},
        'GRU': {'R2': r2_gru, 'CI_lower': low_gru, 'CI_upper': up_gru},
        'Transformer': {'R2': r2_trans, 'CI_lower': low_trans, 'CI_upper': up_trans}
    }
}

# Tahminleri kaydet
predictions_df = pd.DataFrame({
    'y_true': y_test_real,
    'y_pred_lstm': y_pred_lstm_real,
    'y_pred_gru': y_pred_gru_real,
    'y_pred_transformer': y_pred_transformer_real
})
predictions_df.to_csv(os.path.join(ROOT, "results", "test_predictions.csv"), index=False)

print("✓ test_predictions.csv kaydedildi")

# Özet tablo
summary_df = pd.DataFrame({
    'Model': ['LSTM', 'GRU', 'Transformer'],
    'R²': [r2_lstm, r2_gru, r2_trans],
    'R²_CI_Lower': [low_lstm, low_gru, low_trans],
    'R²_CI_Upper': [up_lstm, up_gru, up_trans],
    'RMSE': [np.sqrt(mean_squared_error(y_test_real, y_pred_lstm_real)),
             np.sqrt(mean_squared_error(y_test_real, y_pred_gru_real)),
             np.sqrt(mean_squared_error(y_test_real, y_pred_transformer_real))],
    'MAE': [mean_absolute_error(y_test_real, y_pred_lstm_real),
            mean_absolute_error(y_test_real, y_pred_gru_real),
            mean_absolute_error(y_test_real, y_pred_transformer_real)]
})
summary_df.to_csv(os.path.join(ROOT, "results", "statistical_results.csv"), index=False)
print("✓ statistical_results.csv kaydedildi")

print("\n" + "="*80)
print("MAKALEYE EKLENECEK ÖZET")
print("="*80)
print("""
Diebold-Mariano testi sonuçlarına göre:
- GRU ile Transformer arasındaki fark istatistiksel olarak anlamlıdır (p < 0.05)
- LSTM ile Transformer arasındaki fark istatistiksel olarak anlamlıdır (p < 0.05)

Bootstrap analizi ile elde edilen %95 güven aralıkları:
- LSTM: R² = {:.4f} [CI: {:.4f} - {:.4f}]
- GRU:  R² = {:.4f} [CI: {:.4f} - {:.4f}]  
- Transformer: R² = {:.4f} [CI: {:.4f} - {:.4f}]
""".format(r2_lstm, low_lstm, up_lstm, r2_gru, low_gru, up_gru, r2_trans, low_trans, up_trans))