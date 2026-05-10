import pandas as pd
import os

def load_and_preprocess_data(file_path):
    """
    Veriyi yükler ve Association Rule Mining için Binary (0-1) formata dönüştürür.
    0, 1, 2 puanları -> 0 (Katılmıyor / Düşük risk)
    3, 4 puanları    -> 1 (Katılıyor / Yüksek risk)
    """
    print(f"Veri yükleniyor: {file_path}")
    df = pd.read_csv(file_path, sep=';') # UCI veri seti genelde noktalı virgül ile ayrılır
    
    # Hedef değişkeni (Class: 1 Boşanmış, 0 Evli) ayır
    y = df['Class']
    X = df.drop('Class', axis=1)
    
    # 0-4 arasındaki değerleri 0 ve 1'e dönüştür (Binarization)
    X_binary = X.applymap(lambda x: 1 if x >= 3 else 0)
    
    # İşlenmiş veriyi tekrar birleştir
    processed_df = pd.concat([X_binary, y], axis=1)
    
    # Çıktıyı kaydetmek için klasör kontrolü
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    processed_path = os.path.join(output_dir, 'divorce_processed_binary.csv')
    processed_df.to_csv(processed_path, index=False)
    print(f"Ön işleme tamamlandı. İşlenmiş veri kaydedildi: {processed_path}")
    
    return X_binary, y

if __name__ == "__main__":
    # Test için çalıştır
    # Önce veri setini indirip data/raw/divorce.csv yoluna koyduğundan emin ol
    RAW_DATA_PATH = "data/raw/divorce.csv"
    if os.path.exists(RAW_DATA_PATH):
        load_and_preprocess_data(RAW_DATA_PATH)
    else:
        print(f"HATA: {RAW_DATA_PATH} bulunamadı. Lütfen veri setini ilgili klasöre ekleyin.")