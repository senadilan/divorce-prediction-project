import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
import os

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def load_and_train_models(data_path):
    """Train models and calculate metrics (5 models + ensemble)"""
    
    df = pd.read_csv(data_path)
    y = df['Class']
    X_all = df.drop('Class', axis=1)
    
    important_features = ['Atr1', 'Atr33', 'Atr34', 'Atr35', 'Atr36', 'Atr40', 'Atr51', 'Atr53']
    X_selected = df[important_features]
    
    X_train_all, X_test_all, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_sel, X_test_sel, _, _ = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    xgb_model = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, random_state=42, eval_metric='logloss')
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    lgb_model = LGBMClassifier(n_estimators=100, max_depth=7, learning_rate=0.1, num_leaves=31, random_state=42, verbose=-1)
    
    voting_model = VotingClassifier(
        estimators=[
            ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
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
    
    results = {}
    
    # 54 Features Scenario
    for name, model in models.items():
        model.fit(X_train_all, y_train)
        y_pred = model.predict(X_test_all)
        y_prob = model.predict_proba(X_test_all)[:, 1]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        results[f'{name} (54 Features)'] = {
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
    
    # 8 Features Scenario
    for name, model in models.items():
        model.fit(X_train_sel, y_train)
        y_pred = model.predict(X_test_sel)
        y_prob = model.predict_proba(X_test_sel)[:, 1]
        
        results[f'{name} (8 Features)'] = {
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

def plot_model_comparison_5models(results):
    scenarios = ['SVM (54 Features)', 'XGBoost (54 Features)', 'Random Forest (54 Features)', 
                 'LightGBM (54 Features)', 'Voting Ensemble (54 Features)']
    accuracy_vals = [results[s]['accuracy'] for s in scenarios]
    f1_vals = [results[s]['f1'] for s in scenarios]
    roc_auc_vals = [results[s]['roc_auc'] for s in scenarios]
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, accuracy_vals, width, label='Accuracy', color='#3B82F6')
    ax.bar(x, f1_vals, width, label='F1-Score', color='#10B981')
    ax.bar(x + width, roc_auc_vals, width, label='ROC-AUC', color='#EF4444')
    
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Performance Comparison of 5 Models (54 Features - Test Set)', 
                 pad=20, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.split(' (')[0] for m in scenarios], rotation=15, ha='right')
    ax.set_ylim(0.85, 1.0)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('1_model_comparison_5models.png', dpi=300, bbox_inches='tight')
    print("✓ 1. Model Comparison saved")
    plt.close()

def plot_confusion_matrices_5models(results):
    models = ['SVM (54 Features)', 'XGBoost (54 Features)', 'Random Forest (54 Features)',
              'LightGBM (54 Features)', 'Voting Ensemble (54 Features)']
    
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.flatten()
    
    for idx, model_name in enumerate(models):
        cm = confusion_matrix(results[model_name]['y_test'], results[model_name]['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Married', 'Divorced'],
                    yticklabels=['Married', 'Divorced'],
                    annot_kws={"size": 12, "weight": "bold"})
        
        axes[idx].set_title(f'{model_name.split(" (")[0]}', fontweight='bold')
        axes[idx].set_ylabel('True Label' if idx % 3 == 0 else '')
        axes[idx].set_xlabel('Predicted Label' if idx >= 3 else '')
    
    axes[5].axis('off')
    
    plt.suptitle('Confusion Matrix - 5 Model Comparison (54 Features)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('2_confusion_matrices_5models.png', dpi=300, bbox_inches='tight')
    print("✓ 2. Confusion Matrices saved")
    plt.close()

def plot_scenario_comparison_5models(results):
    models = ['SVM', 'XGBoost', 'Random Forest', 'LightGBM', 'Voting Ensemble']
    scenarios_54 = [f'{m} (54 Features)' for m in models]
    scenarios_8 = [f'{m} (8 Features)' for m in models]
    
    accuracy_54 = [results[s]['accuracy'] for s in scenarios_54]
    accuracy_8 = [results[s]['accuracy'] for s in scenarios_8]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, accuracy_54, width, label='54 Features (All)', color='#8B5CF6')
    ax.bar(x + width/2, accuracy_8, width, label='8 Features (Selected)', color='#EC4899')
    
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Scenario Comparison: 54 Features vs 8 Features (5 Models)', 
                 pad=20, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0.85, 1.0)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(accuracy_54):
        ax.text(i - width/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    for i, v in enumerate(accuracy_8):
        ax.text(i + width/2, v + 0.005, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('3_scenario_comparison_5models.png', dpi=300, bbox_inches='tight')
    print("✓ 3. Scenario Comparison saved")
    plt.close()

def plot_roc_curves_5models(results):
    models = ['SVM (54 Features)', 'XGBoost (54 Features)', 'Random Forest (54 Features)',
              'LightGBM (54 Features)', 'Voting Ensemble (54 Features)']
    colors = ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899']
    
    plt.figure(figsize=(11, 9))
    
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
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('ROC Curve Analysis - 5 Model Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('4_roc_curves_5models.png', dpi=300, bbox_inches='tight')
    print("✓ 4. ROC Curves saved")
    plt.close()

def plot_detailed_metrics_table_5models(results):
    models_to_plot = [
        'SVM (54 Features)', 'XGBoost (54 Features)', 'Random Forest (54 Features)', 'LightGBM (54 Features)', 'Voting Ensemble (54 Features)',
        'SVM (8 Features)', 'XGBoost (8 Features)', 'Random Forest (8 Features)', 'LightGBM (8 Features)', 'Voting Ensemble (8 Features)'
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
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
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
                    colWidths=[0.30, 0.14, 0.14, 0.14, 0.14, 0.14])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    for i in range(6):
        table[(0, i)].set_facecolor('#3B82F6')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, 11):
        for j in range(6):
            if i <= 5:
                table[(i, j)].set_facecolor('#E0F2FE')
            else:
                table[(i, j)].set_facecolor('#FEF3C7')
    
    plt.title('Detailed Metrics Table - 5 Models (Test Set)', 
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('5_metrics_table_5models.png', dpi=300, bbox_inches='tight')
    print("✓ 5. Detailed Metrics Table saved")
    plt.close()

def plot_feature_importance_comparison_extended(data_path):
    df = pd.read_csv(data_path)
    important_features = ['Atr1', 'Atr33', 'Atr34', 'Atr35', 'Atr36', 'Atr40', 'Atr51', 'Atr53']
    X_selected = df[important_features]
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        'XGBoost': XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, random_state=42, eval_metric='logloss'),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'LightGBM': LGBMClassifier(n_estimators=100, max_depth=7, learning_rate=0.1, num_leaves=31, random_state=42, verbose=-1)
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        
        sorted_idx = np.argsort(importances)[::-1]
        sorted_features = [important_features[i] for i in sorted_idx]
        sorted_importances = importances[sorted_idx]
        
        feature_names = {
            'Atr1': 'Lack of Apology',
            'Atr33': 'Criticism',
            'Atr34': 'Aggressiveness',
            'Atr35': 'Insulting',
            'Atr36': 'Humiliating',
            'Atr40': 'Sudden Conflict',
            'Atr51': 'Defensiveness',
            'Atr53': 'Reminding Failures'
        }
        
        display_names = [feature_names.get(f, f) for f in sorted_features]
        colors_palette = ['#8B5CF6', '#EC4899', '#06B6D4', '#10B981', '#F59E0B', '#EF4444', '#6366F1', '#14B8A6']
        
        axes[idx].barh(display_names, sorted_importances, color=colors_palette[:len(sorted_importances)])
        axes[idx].set_xlabel('Importance Score', fontweight='bold')
        axes[idx].set_title(f'{name} - Feature Importance (8 Features)', fontweight='bold', fontsize=12)
        axes[idx].invert_yaxis()
        
        for i, v in enumerate(sorted_importances):
            axes[idx].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.suptitle('Feature Importance Comparison (3 Tree-Based Models)', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('6_feature_importance_comparison_extended.png', dpi=300, bbox_inches='tight')
    print("✓ 6. Feature Importance Comparison saved")
    plt.close()

def plot_association_network():
    G = nx.DiGraph()
    
    edges = [
        ('Atr35\n(Insulting)', 'Atr40\n(Sudden Conflict)', 1.0),
        ('Atr35\n(Insulting)', 'Atr33\n(Criticism)', 1.0),
        ('Atr33\n(Criticism)', 'Atr53\n(Reminding Failures)', 1.0),
        ('Atr1\n(Lack of Apology)', 'Atr53\n(Reminding Failures)', 1.0),
        ('Atr34\n(Aggressiveness)', 'Atr35\n(Insulting)', 0.97),
        ('Atr36\n(Humiliating)', 'Atr35\n(Insulting)', 0.97),
    ]
    
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=1.5)
    
    nx.draw_networkx_nodes(G, pos, node_color='#F87171', node_size=4000, edgecolors='#991B1B')
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='#9CA3AF', width=2)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title('Association Rules: Conflict Cycle Network Map', 
             pad=20, fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('7_psychological_network.png', dpi=300, bbox_inches='tight')
    print("✓ 7. Psychological Network Map saved")
    plt.close()

if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING PRESENTATION QUALITY CHARTS (5 MODELS + ENSEMBLE)...")
    print("=" * 70 + "\n")
    
    PROCESSED_DATA_PATH = "data/processed/divorce_processed_binary.csv"
    
    if os.path.exists(PROCESSED_DATA_PATH):
        print("📊 Training 5 Models...")
        results = load_and_train_models(PROCESSED_DATA_PATH)
        
        print("\n📈 Generating Charts...\n")
        plot_model_comparison_5models(results)
        plot_confusion_matrices_5models(results)
        plot_scenario_comparison_5models(results)
        plot_roc_curves_5models(results)
        plot_detailed_metrics_table_5models(results)
        plot_feature_importance_comparison_extended(PROCESSED_DATA_PATH)
        plot_association_network()
        
        print("\n" + "=" * 70)
        print("✅ ALL CHARTS GENERATED SUCCESSFULLY!")
        print("=" * 70)
    else:
        print(f"ERROR: {PROCESSED_DATA_PATH} not found")
        print("Please run preprocess.py first.")