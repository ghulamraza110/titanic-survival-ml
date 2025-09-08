## 🛳 Titanic Survival Prediction

This project uses machine learning to predict passenger survival on the Titanic, based on demographic and ticketing data. It follows a clean preprocessing pipeline, hyperparameter tuning, and final model deployment for Kaggle submission.

### 📁 Dataset
- **Training data:** `data/train.csv`
- **Test data:** `data/test.csv` (no labels)
- **Submission file:** `data/predictions.csv`

---

### 🔧 Workflow Overview

#### 1. **Exploratory Data Analysis**
- Correlation heatmap using `seaborn`
- Histograms of `Survived` and `Pclass` distributions
- Stratified sampling based on `Survived`, `Pclass`, and `Sex`

#### 2. **Custom Preprocessing Pipeline**
- `AgeImputer`: fills missing age values with mean
- `FeatureEncoder`: one-hot encodes `Embarked` and `Sex`
- `FeatureDropper`: removes irrelevant columns (`Name`, `Ticket`, `Cabin`, etc.)
- `StandardScaler`: normalizes numerical features

#### 3. **Model Training**
- Algorithm: `RandomForestClassifier`
- Hyperparameter tuning via `GridSearchCV`:
  ```python
  param_grid = {
      "n_estimators": [10, 100, 200, 500],
      "max_depth": [None, 5, 10],
      "min_samples_split": [2, 3, 4]
  }
  ```
- Evaluation metric: `accuracy`
- Final model selected using best cross-validation score

#### 4. **Prediction and Submission**
- Preprocessed test data using the same pipeline
- Filled missing values with forward fill (`ffill`)
- Generated predictions with the final model
- Saved results to `predictions.csv` for Kaggle submission

---

### 📊 Model Performance
- **Validation accuracy:** `final_clf.score(X_data_test, y_data_test)`
- **Final model retrained on full training data before prediction**
- **Leaderboard score:** mine Score is 0.77990

---

### 📦 Dependencies
```bash
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

### 🚀 How to Run
```python
# Train model
python train_model.py

# Generate predictions
python predict.py

# Submit predictions to Kaggle
Upload data/predictions.csv to https://www.kaggle.com/c/titanic/submit
```

---

### 🧠 Author Notes
This project demonstrates:
- Custom transformers with `BaseEstimator` and `TransformerMixin`
- Modular pipeline design for reproducibility
- Grid search for model optimization
- Clean separation of training, testing, and production phases

---
