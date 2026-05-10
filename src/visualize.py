import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
import os

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def load_and_train_models(data_path):
    """Modelleri eğit ve metrikleri hesapla"""
    
    df = pd.read_csv(data_path)
    y = df['Class']
    X_all = df.drop('Class', axis=1)
    
    important_features = ['Atr1', 'Atr33', 'Atr34', 'Atr35', 'Atr36', 'Atr40', 'Atr51', 'Atr53']
    X_selected = df[important_features]
    
    # Veri bölüm
    X_train_all, X_test_all, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_sel, X_test_sel, _, _ = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Modeller
    models = {
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'XGBoost': XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, random_state=42, eval_metric='logloss'),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }
    
    results = {}
    
    # 54 Soru Senaryosu
    for name, model in models.items():
        model.fit(X_train_all, y_train)
        y_pred = model.predict(X_test_all)
        y_prob = model.predict_proba(X_test_all)[:, 1]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        results[f'{name} (54 Soru)'] = {
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'model': model
        }
    
    # 8 Soru Senaryosu
    for name, model in models.items():
        model.fit(X_train_sel, y_train)
        y_pred = model.predict(X_test_sel)
        y_prob = model.predict_proba(X_test_sel)[:, 1]
        
        results[f'{name} (8 Soru)'] = {
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'model': model
        }
    
    return results

def plot_model_comparison_3models(results):
    """3 modelin karşılaştırılması (54 soru)"""
    
    scenarios = ['SVM (54 Soru)', 'XGBoost (54 Soru)', 'Random Forest (54 Soru)']
    accuracy_vals = [results[s]['accuracy'] for s in scenarios]
    f1_vals = [results[s]['f1'] for s in scenarios]
    roc_auc_vals = [results[s]['roc_auc'] for s in scenarios]
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width, accuracy_vals, width, label='Accuracy', color='#3B82F6')
    ax.bar(x, f1_vals, width, label='F1-Score', color='#10B981')
    ax.bar(x + width, roc_auc_vals, width, label='ROC-AUC', color='#EF4444')
    
    ax.set_ylabel('Skor', fontweight='bold')
    ax.set_title('3 Modelin Performans Karşılaştırması (54 Soru - Test Seti)', 
                 pad=20, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.split(' (')[0] for m in scenarios])
    ax.set_ylim(0.85, 1.0)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('1_model_comparison_3models.png', dpi=300, bbox_inches='tight')
    print("✓ 1. Model Karşılaştırması (3 model) - kaydedildi")
    plt.close()

def plot_confusion_matrices(results):
    """3 Model için Confusion Matrix"""
    
    models = ['SVM (54 Soru)', 'XGBoost (54 Soru)', 'Random Forest (54 Soru)']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, model_name in enumerate(models):
        cm = confusion_matrix(results[model_name]['y_test'], results[model_name]['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Evli Kalacak', 'Boşanacak'],
                    yticklabels=['Evli Kalacak', 'Boşanacak'],
                    annot_kws={"size": 12, "weight": "bold"})
        
        axes[idx].set_title(f'{model_name.split(" (")[0]}', fontweight='bold')
        axes[idx].set_ylabel('Gerçek' if idx == 0 else '')
        axes[idx].set_xlabel('Tahmin' if idx == 1 else '')
    
    plt.suptitle('Confusion Matrix - 3 Model Karşılaştırması (54 Soru)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('2_confusion_matrices_3models.png', dpi=300, bbox_inches='tight')
    print("✓ 2. Confusion Matrix (3 Model) - kaydedildi")
    plt.close()

def plot_scenario_comparison(results):
    """54 Soru vs 8 Soru Karşılaştırması"""
    
    models = ['SVM', 'XGBoost', 'Random Forest']
    scenarios_54 = [f'{m} (54 Soru)' for m in models]
    scenarios_8 = [f'{m} (8 Soru)' for m in models]
    
    accuracy_54 = [results[s]['accuracy'] for s in scenarios_54]
    accuracy_8 = [results[s]['accuracy'] for s in scenarios_8]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, accuracy_54, width, label='54 Soru (Tam)', color='#8B5CF6')
    ax.bar(x + width/2, accuracy_8, width, label='8 Soru (Seçilmiş)', color='#EC4899')
    
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Senaryo Karşılaştırması: 54 Soru vs 8 Soru', 
                 pad=20, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.85, 1.0)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(accuracy_54):
        ax.text(i - width/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    for i, v in enumerate(accuracy_8):
        ax.text(i + width/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('3_scenario_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 3. Senaryo Karşılaştırması (54 vs 8 Soru) - kaydedildi")
    plt.close()

def plot_roc_curves(results):
    """ROC Eğrileri - 3 Model"""
    
    models = ['SVM (54 Soru)', 'XGBoost (54 Soru)', 'Random Forest (54 Soru)']
    colors = ['#3B82F6', '#10B981', '#F59E0B']
    
    plt.figure(figsize=(10, 8))
    
    for model_name, color in zip(models, colors):
        y_test = results[model_name]['y_test']
        y_prob = results[model_name]['y_prob']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, lw=2.5,
                label=f'{model_name.split(" (")[0]} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Yanlış Alarm)', fontweight='bold')
    plt.ylabel('True Positive Rate (Doğru Tespit)', fontweight='bold')
    plt.title('ROC Eğrisi Analizi - 3 Model Karşılaştırması', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('4_roc_curves_3models.png', dpi=300, bbox_inches='tight')
    print("✓ 4. ROC Eğrileri (3 Model) - kaydedildi")
    plt.close()

def plot_detailed_metrics_table(results):
    """Detaylı metrik tablosu"""
    
    models_to_plot = [
        'SVM (54 Soru)', 'XGBoost (54 Soru)', 'Random Forest (54 Soru)',
        'SVM (8 Soru)', 'XGBoost (8 Soru)', 'Random Forest (8 Soru)'
    ]
    
    data = []
    for model_name in models_to_plot:
        data.append({
            'Model': model_name,
            'Accuracy': results[model_name]['accuracy'],
            'Precision': results[model_name]['precision'],
            'Recall': results[model_name]['recall'],
            'F1-Score': results[model_name]['f1'],
            'ROC-AUC': results[model_name]['roc_auc']
        })
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')
    
    # Renk formatı
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['Model'],
            f"{row['Accuracy']:.4f}",
            f"{row['Precision']:.4f}",
            f"{row['Recall']:.4f}",
            f"{row['F1-Score']:.4f}",
            f"{row['ROC-AUC']:.4f}"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Header stilini ayarla
    for i in range(6):
        table[(0, i)].set_facecolor('#3B82F6')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Satır rengini ayarla
    for i in range(1, 7):
        for j in range(6):
            if i <= 3:
                table[(i, j)].set_facecolor('#E0F2FE')
            else:
                table[(i, j)].set_facecolor('#FEF3C7')
    
    plt.title('Detaylı Metrik Tablosu - Tüm Modeller (Test Seti)', 
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('5_metrics_table.png', dpi=300, bbox_inches='tight')
    print("✓ 5. Detaylı Metrik Tablosu - kaydedildi")
    plt.close()

def plot_feature_importance_comparison(data_path):
    """Feature Importance - XGBoost ve Random Forest Karşılaştırması"""
    
    df = pd.read_csv(data_path)
    important_features = ['Atr1', 'Atr33', 'Atr34', 'Atr35', 'Atr36', 'Atr40', 'Atr51', 'Atr53']
    X_selected = df[important_features]
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        'XGBoost': XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, random_state=42, eval_metric='logloss'),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        
        # Sırala
        sorted_idx = np.argsort(importances)[::-1]
        sorted_features = [important_features[i] for i in sorted_idx]
        sorted_importances = importances[sorted_idx]
        
        # Özel isimler
        feature_names = {
            'Atr1': 'Özür Dilememe',
            'Atr33': 'Kişiliğe Eleştiri',
            'Atr34': 'Saldırganlık',
            'Atr35': 'Hakaret',
            'Atr36': 'Aşağılama',
            'Atr40': 'Ani Kavga',
            'Atr51': 'Savunmacılık',
            'Atr53': 'Eksik Yüze Vurma'
        }
        
        display_names = [feature_names.get(f, f) for f in sorted_features]
        
        axes[idx].barh(display_names, sorted_importances, color=['#8B5CF6', '#EC4899', '#06B6D4', '#10B981', '#F59E0B', '#EF4444', '#6366F1', '#14B8A6'][:len(sorted_importances)])
        axes[idx].set_xlabel('Önem Oranı', fontweight='bold')
        axes[idx].set_title(f'{name} - Öznitelik Önemi (8 Soru)', fontweight='bold', fontsize=12)
        axes[idx].invert_yaxis()
        
        # Değerleri göster
        for i, v in enumerate(sorted_importances):
            axes[idx].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.suptitle('Feature Importance Karşılaştırması (XGBoost vs Random Forest)', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('6_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 6. Feature Importance Karşılaştırması - kaydedildi")
    plt.close()

def plot_association_network():
    """Psikolojik Ağ Haritası (Değiştirilmemiş)"""
    
    G = nx.DiGraph()
    
    edges = [
        ('Atr35\n(Hakaret)', 'Atr40\n(Ani Kavga)', 1.0),
        ('Atr35\n(Hakaret)', 'Atr33\n(Kişiliğe Eleştiri)', 1.0),
        ('Atr33\n(Kişiliğe Eleştiri)', 'Atr53\n(Eksik Yüze Vurma)', 1.0),
        ('Atr1\n(Özür Dilememe)', 'Atr53\n(Eksik Yüze Vurma)', 1.0),
        ('Atr34\n(Saldırganlık)', 'Atr35\n(Hakaret)', 0.97),
        ('Atr36\n(Aşağılama)', 'Atr35\n(Hakaret)', 0.97),
    ]
    
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=1.5)
    
    nx.draw_networkx_nodes(G, pos, node_color='#F87171', node_size=4000, edgecolors='#991B1B')
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='#9CA3AF', width=2)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title('Birliktelik Kuralları: Çatışma Döngüsü Ağ Haritası', 
             pad=20, fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('7_psychological_network.png', dpi=300, bbox_inches='tight')
    print("✓ 7. Psikolojik Ağ Haritası - kaydedildi")
    plt.close()

if __name__ == "__main__":
    print("=" * 70)
    print("SUNUM KALİTESİNDE GRAFIKLER OLUŞTURULUYOR...")
    print("=" * 70 + "\n")
    
    PROCESSED_DATA_PATH = "data/processed/divorce_processed_binary.csv"
    
    if os.path.exists(PROCESSED_DATA_PATH):
        # Modelleri eğit
        print("📊 Modeller eğitiliyor...")
        results = load_and_train_models(PROCESSED_DATA_PATH)
        
        # Grafikler oluştur
        print("\n📈 Grafikler oluşturuluyor...\n")
        plot_model_comparison_3models(results)
        plot_confusion_matrices(results)
        plot_scenario_comparison(results)
        plot_roc_curves(results)
        plot_detailed_metrics_table(results)
        plot_feature_importance_comparison(PROCESSED_DATA_PATH)
        plot_association_network()
        
        print("\n" + "=" * 70)
        print("✅ TÜM GRAFİKLER BAŞARIYLA OLUŞTURULDU!")
        print("=" * 70)
    else:
        print(f"HATA: {PROCESSED_DATA_PATH} bulunamadı")
        print("Lütfen önce preprocess.py dosyasını çalıştırın.")