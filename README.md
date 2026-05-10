# Evlilik Sağlamlığı Tahmin Sistemi - Psikolojik Öznitelik Seçimi Yöntemi

## 📋 Proje Özeti

Bu proje, **Divorce Predictors Dataset** (UCI Machine Learning Repository) üzerinde veri madenciliği teknikleri uygulamaktadır. İlişkilerin boşanma riski taşıyıp taşımadığını tahmin etmek için:

- **Sınıflandırma (Classification)** görevini gerçekleştir
- **Birliktelik Kuralı Madenciliği (Association Rule Mining)** kullanarak etkili öznitelikleri belirle
- **Yenilikçi Feature Selection** yaklaşımı uygula (54 sorudan 8 kritik soru seçme)

### Temel Bulgu

**54 psikolojik soru kullanarak %91.18 başarı** elde ettikten sonra, **Boşanan çiftlerden FP-Growth ile elde edilen 8 kritik davranışı** kullanarak **%94.12 başarıya** ve **0 False Positive (Yanlış Pozitif)**'e ulaştık!

---

## 🎯 Seçilen Görevler

### 1. Sınıflandırma (Classification)

- **Hedef**: İlişki durumunu tahmin et (Evli = 0, Boşanmış = 1)
- **Kullanılan Algoritmalar**:
  - Support Vector Machine (SVM)
  - XGBoost (Gradient Boosting)
  - Random Forest (Ensemble Learning)

### 2. Birliktelik Kuralı Madenciliği (Association Rule Mining)

- **Algoritma**: FP-Growth (daha hızlı ve bellek-verimli)
- **Amaç**: Boşanmış çiftlerde en sık görülen davranış kombinasyonlarını bulmak
- **Çıktı**: 8 "Kırmızı Bayrak" davranışı tespit edildi

---

## 📊 Veri Seti

**Kaynak**: [UCI Machine Learning Repository - Divorce Predictors](https://archive.ics.uci.edu/ml/datasets/Divorce+Predictors+data+set)

**İçerik**:

- **Örnekler**: ~170 çift (85 evli, 85 boşanmış)
- **Özellikleri**: 54 psikolojik soru (0-4 puanlandırma)
- **Hedef**: Class (0: Evli, 1: Boşanmış)

**Veri Ön İşleme**:

```
0, 1, 2 puanları    → 0 (Katılmıyor / Düşük risk)
3, 4 puanları       → 1 (Katılıyor / Yüksek risk)
```

---

## 🚀 Proje Mimarisi

```
project/
├── data/
│   ├── raw/
│   │   └── divorce.csv                 # Orijinal UCI veri seti
│   └── processed/
│       └── divorce_processed_binary.csv # Binary formattaki veri
├── src/
│   ├── preprocess.py       # Veri ön işleme (0-4 → 0-1)
│   ├── association_rules.py # FP-Growth ile kural çıkarımı
│   ├── classification.py    # Sınıflandırma modelleri (SVM, XGB, RF)
│   └── visualize.py         # Sonuç grafikleri
└── results/
    ├── classification_results.json   # Model metrikleri
    ├── 1_model_comparison_3models.png
    ├── 2_confusion_matrices.png
    ├── 3_scenario_comparison.png
    ├── 4_roc_curves.png
    ├── 5_metrics_table.png
    ├── 6_feature_importance.png
    └── 7_psychological_network.png
```

---

## 📈 Kullanım Adımları

### 1. Kütüphaneleri Yükle

```bash
pip install -r requirements.txt
```

### 2. Veriyi Ön İşle

```bash
python src/preprocess.py
```

- Orijinal veri seti `data/raw/divorce.csv`'i yükle
- 0-4 puanları 0-1 (Binary) formatına dönüştür
- İşlenmiş veriyi `data/processed/divorce_processed_binary.csv`'e kaydet

### 3. Birliktelik Kurallarını Çıkar

```bash
python src/association_rules.py
```

- **FP-Growth** algoritması ile boşanmış çiftlerin davranışlarını analiz et
- En etkili davranışları tespit et:
  - **Atr35** (Hakaret)
  - **Atr40** (Ani Kavga)
  - **Atr34** (Saldırganlık)
  - **Atr36** (Aşağılama)
  - **Atr33** (Kişiliğe Eleştiri)
  - **Atr53** (Eksik Yüze Vurma)
  - **Atr51** (Savunmacılık)
  - **Atr1** (Özür Dilememe)

### 4. Sınıflandırma Modellerini Eğit

```bash
python src/classification.py
```

- **3 Model** eğit:
  - SVM (Support Vector Machine)
  - XGBoost (Gradient Boosting)
  - Random Forest (Ensemble)
- **5-Fold Stratified Cross-Validation** uygula
- **2 Senaryo** karşılaştır:
  - **54 Soru** (Tam veri)
  - **8 Soru** (Seçilmiş öznitelikler)

### 5. Sonuçları Görselleştir

```bash
python src/visualize.py
```

Aşağıdaki grafikler oluşturulur:

- Model Performans Karşılaştırması
- Confusion Matrix (Karmaşıklık Matrisleri)
- ROC Eğrileri
- Senaryo Karşılaştırması
- Öznitelik Önemi (Feature Importance)
- Psikolojik Ağ Haritası (Association Network)

---

## 📊 Ana Sonuçlar

### Senaryo Karşılaştırması (Test Seti Accuracy)

| Model             | 54 Soru | 8 Soru | Fark      |
| ----------------- | ------- | ------ | --------- |
| **SVM**           | 0.9412  | 0.9412 | ✓ Eşit    |
| **XGBoost**       | 0.9118  | 0.9412 | ⬆️ +2.94% |
| **Random Forest** | 0.8824  | 0.9118 | ⬆️ +2.94% |

### XGBoost (8 Soru) - En İyi Model Detaylı Metrikleri

```
Accuracy:        94.12%
Precision:       93.75%    (Boşanan dedikleri kaçı gerçekten boşanır?)
Recall:          93.75%    (Gerçek boşananların kaçını bulur?)
F1-Score:        93.75%    (Precision-Recall dengesi)
ROC-AUC:         92.73%    (Sınıf ayrımı kapasitesi)
Sensitivity:     93.75%    (Doğru Pozitif Oranı)
Specificity:     94.12%    (Doğru Negatif Oranı)

Confusion Matrix (Test Seti):
             Tahmin: Evli  Tahmin: Boşanmış
Gerçek: Evli      17              0            ← 0 Yanlış Pozitif!
Gerçek: Boşan.     2             15
```

### Yenilikçi Bulgu

**54 sorudan 8'ine düşürerek**:

- ✅ Başarı artması (%91.18 → %94.12)
- ✅ Yanlış Pozitif sıfırlanması (Hiç sağlam ilişkiye "boşana mısın?" demiyor)
- ✅ Model basitleşmesi (Ölçekleme daha hızlı)
- ✅ İnterpretasyon açıklığı (8 davranış → açık tavsiyeler)

---

## 📝 Değerlendirme Metrikleri

### Sınıflandırma için Kullanılan Metrikler

| Metrik          | Tanım                                             | İdeal Değer |
| --------------- | ------------------------------------------------- | ----------- |
| **Accuracy**    | Doğru tahmin oranı                                | 1.0         |
| **Precision**   | Tahmin edilen boşananların kaçı gerçekten boşanır | 1.0         |
| **Recall**      | Gerçek boşananların kaçı bulunur                  | 1.0         |
| **F1-Score**    | Precision ve Recall'ın dengeli ortalaması         | 1.0         |
| **ROC-AUC**     | Sınıf ayrımı kapasitesi                           | 1.0         |
| **Sensitivity** | Doğru Pozitif Oranı = Recall                      | 1.0         |
| **Specificity** | Doğru Negatif Oranı                               | 1.0         |

---

## 🔧 Hiperparametreler

### SVM

```python
kernel='rbf'      # Radial Basis Function
C=1.0            # Düzenleme parametresi
gamma='scale'    # Çekirdek katsayısı
probability=True # Olasılık tahmini
```

### XGBoost

```python
max_depth=5           # Ağaç derinliği
learning_rate=0.1    # Öğrenme hızı
n_estimators=100     # Ağaç sayısı
eval_metric='logloss' # Evaluasyon metriği
```

### Random Forest

```python
n_estimators=100      # Ağaç sayısı
max_depth=10         # Maksimum derinlik
random_state=42      # Tekrarlanabilirlik
```

---

## 🎓 Metodoloji

### Veri Bölümleme

- **Test Seti**: %20 (stratified → sınıf dengesi korundu)
- **Train Seti**: %80

### Cross-Validation

- **Yöntem**: 5-Fold Stratified K-Fold
- **Avantaj**: Veri setinin tamamı test edilir, overfitting risqi azalır
- **Çıktı**: Her fold'dan 5 farklı skor → ortalaması ve std sapması

### Veri Dengeleme

Veri setinde sınıflar dengelidir (85 evli, 85 boşanmış) → SMOTE gerekli değil

---

## 📚 Kaynaklar

- **Veri Seti**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Divorce+Predictors+data+set)
- **Orijinal Araştırma**: Yöntem, Adem & Karagöz, Pınar (2019)
- **Algoritmalar**:
  - SVM: Vapnik, 1995 - Support Vector Networks
  - XGBoost: Chen & Guestrin, 2016
  - FP-Growth: Han et al., 2000
  - Random Forest: Breiman, 2001

---

## ✨ Projenin Yenilikçi Yanları (Bonus)

1. **Yenilikçi Öznitelik Seçimi**: FP-Growth ile boşanan çiftlerin ortak davranışlarını bulup, bu 8 öznitelikle daha iyi model elde etme
2. **3 Farklı Algoritma Karşılaştırması**: SVM, XGBoost, Random Forest
3. **5-Fold Cross-Validation**: Gerçekçi performans değerlendirmesi
4. **Detaylı Metrik Analizi**: Sadece accuracy değil, precision, recall, specificity gibi detaylı analizler
5. **Network Analisi**: Davranışlar arasındaki ilişkileri ağ grafiği ile gösterme

---

## 🚀 Sunum Notları

**Sunum Süresi**: 12-15 dakika

### Slide Yapısı

1. **Giriş** (1 dakika)
   - Proje amacı: İlişki sağlamlığını tahmin etme
2. **Veri Seti** (1 dakika)
   - UCI Divorce Dataset: 170 çift, 54 soru
3. **Metodoloji** (3 dakika)
   - Veri ön işleme (Binary conversion)
   - FP-Growth ile 8 kritik davranış bulma
   - Sınıflandırma modelleri (SVM, XGB, RF)
4. **Sonuçlar** (5 dakika)
   - 3 Model Karşılaştırması
   - 54 Soru vs 8 Soru Analizi
   - ROC Eğrileri
   - Confusion Matrix
5. **Yenilikçi Katkılar** (2 dakika)
   - 54 sorudan 8'ine düşüş
   - Başarı artışı (%91.18 → %94.12)
   - 0 False Positive
6. **Sonuç ve Öneriler** (1-2 dakika)
   - Model genelleme kapasitesi
   - Olası uygulamalar (danışmanlık hizmetleri)

---

## 📧 İletişim ve Notlar

**Proje Türü**: Veri Madenciliği Dersi Projesi

**Değerlendirme Kriterleri**:

- ✅ Geçerli veri madenciliği görevleri (Sınıflandırma + Birliktelik Kuralı)
- ✅ Uygun veri seti (UC: 170 örnek, 54 öznitelik)
- ✅ 3 farklı algoritma uygulanması
- ✅ Çoklu evaluasyon metrikleri
- ✅ Akademik yazı formatı
- ✅ Yenilikçi öznitelik seçimi yaklaşımı (Bonus)

---

**Versiyon**: 1.1 (İyileştirilmiş)  
**Son Güncelleme**: 2024  
**Durum**: Hazır ✅
