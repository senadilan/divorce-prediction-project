

💔 AI-Driven Divorce Prediction: A Hybrid Data Mining Approach


📌 Project Overview
This repository contains a comprehensive data mining and machine learning pipeline designed to predict marital stability and divorce risks based on psychological behavioral patterns.

Traditional psychometric evaluations often rely on exhaustive questionnaires containing redundant noise. This project introduces a Hybrid Feature Selection Approach, leveraging Association Rule Mining (FP-Growth) to isolate 8 critical behavioral "Red Flags" from a 54-question dataset. These isolated features are then fed into an advanced Soft Voting Ensemble to predict relationship outcomes with high accuracy and zero false positives.

🧠 Methodology & Pipeline
Data Preprocessing (Binarization): 5-point Likert scale answers were binarized to prepare the dataset for transactional rule mining.

Association Rule Mining (FP-Growth): Extracted concurrent toxic behaviors specific to divorced couples, reducing the feature space from 54 to 8 core predictors (e.g., Sudden Conflict, Insulting, Defensiveness).

Classification & Ensemble Learning: Evaluated the dataset using 5 models (SVM, XGBoost, Random Forest, LightGBM, Voting Ensemble) combined with 5-Fold Stratified Cross-Validation.

📊 Dataset
Source: UCI Machine Learning Repository - Divorce Predictors Data Set

Instances: 170 (85 Divorced, 85 Married)

Attributes: 54 psychological metrics based on Gottman Couples Therapy.

🏆 Key Results
By reducing the dimensionality from 54 features to the critical 8 "Red Flags", the boosting models became more stable. The Voting Ensemble achieved the following metrics on the test set:

Accuracy: 94.12%

Precision: 1.0000 (0 False Positives - The model never misdiagnosed a stable marriage)

ROC-AUC: 0.9446


## 📊 Results


| Model             | 54 Question | 8 Question|     
| ----------------- | -------     | ------    | 
| **SVM**           | 0.9412      | 0.9412    |
| **XGBoost**       | 0.9118      | 0.9412    |  
| **Random Forest** | 0.8824      | 0.9118    |


🚀 Installation & Usage
1. Clone the repository:

git clone https://github.com/yourusername/divorce-prediction-ai.git
cd divorce-prediction-ai

2. Install required dependencies:

pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn networkx

3. Run the pipeline sequentially:

python src/preprocess.py

python src/association_rules.py
python src/classification.py

python src/visualize.py

📁 Project Structure
data/

raw/divorce.csv (Original UCI dataset)

processed/divorce_processed.csv (Binarized dataset)

src/

preprocess.py

association_rules.py

classification.py

visualize.py

docs/

presentation.html (HTML Presentation slides)

IEEE_Paper.pdf (Academic research paper format)

README.md

👥 Authors
Sena Dilan Çakır - Software Engineering Student, AYBÜ

Rojda Süslü - Software Engineering Student, AYBÜ

⚖️ Ethical Considerations
This predictive engine is designed purely for academic research and secondary decision-support for human professionals. Due to the high sensitivity of marital data and potential cultural biases in psychological expressions, this model should not be utilized as an autonomous diagnostic toolset.
---

**Son Güncelleme**: 2024  
**Durum**: Hazır ✅
