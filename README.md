# Heart Disease Prediction with XGBoost

Projekt przedstawia kompletny pipeline uczenia maszynowego do predykcji występowania choroby serca (`HeartDisease`)
na podstawie danych klinicznych. Wykorzystano model **XGBoost** oraz pełny preprocessing
(imputacja, standaryzacja i One‑Hot Encoding) w jednym pipeline, aby uniknąć data leakage.
Celem jest maksymalizacja jakości wykrywania pacjentów wysokiego ryzyka przy zachowaniu dobrej generalizacji.

> **Uwaga**: Model powstał na danych publicznych i nie jest narzędziem diagnostycznym.
> Służy wyłącznie do celów edukacyjnych/analitycznych.

## Dane
Dataset: `heart.csv` (Heart Failure Prediction Dataset, Kaggle).  
Zmienna docelowa: `HeartDisease` (0/1).

Podział danych:
- 70% train / 30% test
- Stratyfikacja klas
- `random_state=42` dla powtarzalności

## Metodyka
1. **Preprocessing**
   - zmienne numeryczne: imputacja średnią + standaryzacja  
   - zmienne kategoryczne: imputacja dominantą + One‑Hot Encoding
2. **Model**
   - `XGBClassifier`
   - strojenie hiperparametrów: `RandomizedSearchCV` (5‑fold Stratified CV)
3. **Ewaluacja**
   - ROC AUC, PR AUC
   - precision, recall, F1, balanced accuracy
   - macierz pomyłek
   - interpretowalność: feature importance + SHAP (w notebooku)

## Wyniki (skrót)
Najlepszy model osiąga na zbiorze testowym (wartości mogą się minimalnie różnić między uruchomieniami):
- Accuracy ~0.88  
- ROC AUC ~0.93  
- Recall klasy pozytywnej ~0.88  

Pełne metryki oraz wykresy ROC/PR i SHAP znajdują się w notebooku.

## Jak uruchomić
```bash
pip install -r requirements.txt
python src/train.py --data data/heart.csv --out models/model.joblib
python src/evaluate.py --data data/heart.csv --model models/model.joblib
```
Notebook:
```bash
jupyter notebook notebooks/01_xgboost_heart_disease.ipynb
```

## Struktura repo
```
heart-disease-xgboost/
├── data/heart.csv
├── notebooks/01_xgboost_heart_disease.ipynb
├── src/train.py
├── src/evaluate.py
├── requirements.txt
└── README.md
```

## Ograniczenia
- Dane są publiczne, zebrane w różnych ośrodkach i nie reprezentują pełnej populacji.
- Model nie może być stosowany do rzeczywistych decyzji medycznych.
