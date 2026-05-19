import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier  # ← YENİ
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, make_scorer
)
import warnings
import os
import json

warnings.filterwarnings("ignore", category=FutureWarning)

def evaluate_model(y_true, y_pred, y_prob, model_name=""):
    """Detaylı model değerlendirme metrikleri"""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }
    
    # Ek metrikler
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["True Negatives"] = int(tn)
    metrics["False Positives"] = int(fp)
    metrics["False Negatives"] = int(fn)
    metrics["True Positives"] = int(tp)
    metrics["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics["Sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics

def run_classification_with_cv(data_path, output_dir="results"):
    """
    5-Fold Stratified CV ile sınıflandırma eğitimi
    SVM, XGBoost, Random Forest, LightGBM modellerini ve Voting Ensemble'ı karşılaştırır
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("=" * 70)
    print("BOŞANMA TAHMİN SİSTEMİ - İYİLEŞTİRİLMİŞ PIPELINE (5 MODEL + ENSEMBLE)")
    print("=" * 70)
    
    # Veri yükleme
    print("\n[1/5] Veri yükleniyor...")
    try:
        df = pd.read_csv(data_path)
        print(f"✓ {len(df)} örnek yüklendi")
    except FileNotFoundError:
        print(f"✗ HATA: {data_path} bulunamadı")
        return
    
    y = df['Class']
    X_all = df.drop('Class', axis=1)
    
    # FP-Growth'tan bulduğumuz 8 kritik öznitelik
    important_features = ['Atr1', 'Atr33', 'Atr34', 'Atr35', 'Atr36', 'Atr40', 'Atr51', 'Atr53']
    X_selected = df[important_features]
    
    print(f"✓ 54 soru (tam) ve 8 soru (seçilmiş) özellikleri ayarlandı")
    
    # Final test seti (CV'nin dışında tutulacak)
    print("\n[2/5] Test seti ayrılıyor (%20)...")
    X_train_all, X_test_all, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_sel, X_test_sel, _, _ = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Train: {len(X_train_all)} | Test: {len(X_test_all)}")
    
    # ===== MODELLER - YENİ: LightGBM ve Voting Ensemble =====
    print("\n[3/5] Modeller tanımlanıyor...")
    
    # Bireysel modeller
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    xgb_model = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, random_state=42, eval_metric='logloss')
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    lgb_model = LGBMClassifier(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    
    # Voting Ensemble - tüm modelleri birleştir
    voting_model = VotingClassifier(
        estimators=[
            ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)),
            ('xgb', XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, random_state=42, eval_metric='logloss')),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
            ('lgb', LGBMClassifier(n_estimators=100, max_depth=7, learning_rate=0.1, num_leaves=31, random_state=42, verbose=-1))
        ],
        voting='soft'
    )
    
    models = {
        'SVM': svm_model,
        'XGBoost': xgb_model,
        'Random Forest': rf_model,
        'LightGBM': lgb_model,
        'Voting Ensemble': voting_model
    }
    
    print(f"✓ {len(models)} model hazır: {list(models.keys())}")
    
    # Sonuçları sakla
    cv_results = {}
    final_results = {}
    feature_importances = {}
    
    # 5-Fold CV Evaluation
    print("\n[4/5] 5-Fold Stratified Cross-Validation çalışıyor...")
    print("-" * 70)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for scenario in ['54 Soru', '8 Soru']:
        print(f"\n📊 SENARYO: {scenario}")
        print("=" * 70)
        
        X_train = X_train_all if scenario == '54 Soru' else X_train_sel
        X_test = X_test_all if scenario == '54 Soru' else X_test_sel
        
        for model_name, model in models.items():
            print(f"\n  Model: {model_name}")
            
            # CV skoru hesapla
            scoring = {
                'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score, zero_division=0),
                'recall': make_scorer(recall_score, zero_division=0),
                'f1': make_scorer(f1_score, zero_division=0),
                'roc_auc': make_scorer(roc_auc_score)
            }
            
            cv_scores = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring, return_train_score=False)
            
            # Ortalama ve std göster
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                mean_score = cv_scores[f'test_{metric}'].mean()
                std_score = cv_scores[f'test_{metric}'].std()
                print(f"    {metric.upper():<10}: {mean_score:.4f} ± {std_score:.4f}")
            
            # Sonuçları sakla
            key = f"{model_name} ({scenario})"
            cv_results[key] = {
                metric: cv_scores[f'test_{metric}'] for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            }
        
        # Final test seti tahmini
        print(f"\n{'─' * 70}")
        print(f"TEST SETİ SONUÇLARI:")
        print(f"{'─' * 70}")
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            metrics = evaluate_model(y_test, y_pred, y_prob, model_name)
            key = f"{model_name} ({scenario})"
            final_results[key] = metrics
            
            print(f"\n  {model_name}:")
            print(f"    Accuracy:  {metrics['Accuracy']:.4f}")
            print(f"    Precision: {metrics['Precision']:.4f}")
            print(f"    Recall:    {metrics['Recall']:.4f}")
            print(f"    F1-Score:  {metrics['F1-Score']:.4f}")
            print(f"    ROC-AUC:   {metrics['ROC-AUC']:.4f}")
            print(f"    Sensitivity (TPR): {metrics['Sensitivity']:.4f}")
            print(f"    Specificity (TNR): {metrics['Specificity']:.4f}")
            print(f"    False Positives: {int(metrics['False Positives'])}")
            print(f"    False Negatives: {int(metrics['False Negatives'])}")
            
            # Feature importance (uygun olanlarda)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = X_test.columns
                feature_importances[key] = dict(zip(feature_names, importances))
    
    # Sonuçları JSON'a kaydet
    print("\n[5/5] Sonuçlar kaydediliyor...")
    
    # float32 → float dönüştürme fonksiyonu
    def convert_to_serializable(obj):
        """NumPy veri tiplerini JSON-serializable yap"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results_summary = {
        "cv_results": {k: {m: v.tolist() for m, v in cv.items()} for k, cv in cv_results.items()},
        "test_results": convert_to_serializable(final_results),
        "feature_importances": convert_to_serializable(feature_importances)
    }
    
    output_path = os.path.join(output_dir, "classification_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✓ Sonuçlar kaydedildi: {output_path}")
    
    # Özet tablosu
    print("\n" + "=" * 70)
    print("ÖZET: TEST SETİ ACCURACY KARŞILAŞTIRMASI (5 MODEL + ENSEMBLE)")
    print("=" * 70)
    summary_df = pd.DataFrame({
        model: [final_results[f"{model} (54 Soru)"]["Accuracy"],
                final_results[f"{model} (8 Soru)"]["Accuracy"]]
        for model in models.keys()
    }, index=['54 Soru', '8 Soru'])
    
    print(summary_df.round(4))
    print("\n✓ Pipeline başarıyla tamamlandı!")
    
    return final_results, feature_importances

if __name__ == "__main__":
    PROCESSED_DATA_PATH = "data/processed/divorce_processed_binary.csv"
    
    if os.path.exists(PROCESSED_DATA_PATH):
        results, importances = run_classification_with_cv(PROCESSED_DATA_PATH)
    else:
        print(f"HATA: İşlenmiş veri bulunamadı: {PROCESSED_DATA_PATH}")
        print("Lütfen önce preprocess.py dosyasını çalıştırın.")