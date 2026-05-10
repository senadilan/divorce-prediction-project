import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

def extract_rules(processed_file_path, min_support=0.8, min_confidence=0.9):
    """
    İşlenmiş veriyi kullanarak FP-Growth algoritması ile birliktelik kurallarını çıkarır.
    Apriori'ye göre çok daha bellek dostudur.
    """
    print("Birliktelik kuralları FP-Growth ile analiz ediliyor...")
    df = pd.read_csv(processed_file_path)
    
    # Sadece 'Class' 1 olan (Boşanmış) çiftleri analiz edelim
    divorced_df = df[df['Class'] == 1].drop('Class', axis=1)
    
    # mlxtend kütüphanesinin memory hatasını ve warning'ini çözmek için veriyi bool yapıyoruz
    divorced_df = divorced_df.astype(bool)
    
    # FP-Growth kullanıyoruz. 
    # min_support=0.8: Boşanmış çiftlerin en az %80'inde görülmeli
    # max_len=3: Maksimum 3'lü kombinasyonlara bak (63GB RAM patlamasını önler)
    frequent_itemsets = fpgrowth(divorced_df, min_support=min_support, use_colnames=True, max_len=3)
    
    # Birliktelik kurallarını oluştur
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    # Kuralları "Lift" (kaldıraç/etki) değerine göre büyükten küçüğe sırala
    rules = rules.sort_values(by='lift', ascending=False)
    
    print(f"Toplam {len(rules)} kural bulundu.")
    return rules

if __name__ == "__main__":
    # Yolu direkt ana klasörden başlatıyoruz
    PROCESSED_DATA_PATH = "data/processed/divorce_processed_binary.csv"
    try:
        rules_df = extract_rules(PROCESSED_DATA_PATH)
        
        print("\n--- EN TEHLİKELİ DAVRANIŞ KOMBİNASYONLARI (İLK 5) ---")
        for index, row in rules_df.head(5).iterrows():
            antecedents = list(row['antecedents'])
            consequents = list(row['consequents'])
            print(f"Eğer {antecedents} görülüyorsa -> {consequents} de görülür (Güven: %{row['confidence']*100:.1f})")
            
    except FileNotFoundError:
        print("HATA: İşlenmiş veri bulunamadı. Önce preprocess.py dosyasını çalıştırın.")