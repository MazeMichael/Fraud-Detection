# Fraud-Detection

# Fraud Detection with Random Forest

This project uses a Random Forest classifier to detect potentially fraudulent transactions in a synthetic financial dataset. The goal is to predict whether a transaction is fraudulent (`isFraud`) based on features like transaction type, amount, and other metadata.

---

## Dataset

The dataset used is a synthetic financial transactions log, originally sourced from [Kaggle or OpenML].
- Download here: https://www.kaggle.com/datasets/sriharshaeedala/financial-fraud-detection-dataset?resource=download 
- `nameOrig` and `nameDest`: identifiers for source and destination accounts  
- `type`: type of transaction (e.g., TRANSFER, CASH_OUT)  
- `amount`, `oldbalanceOrg`, `newbalanceOrig`, etc.: transaction and account balance details  
- `isFraud`: **target variable** (1 = fraud, 0 = not fraud)  
- `isFlaggedFraud`: dropped for the purposes of this model

---

## Project Workflow

1. **Preprocessing**
   - `nameOrig` and `nameDest` (string fields) are stored separately and dropped from modeling
   - One-hot encoding is applied to the `type` column using `ColumnTransformer` and `OneHotEncoder`
   - Features (`X`) and target (`y`) are extracted
   - Data is split into training and testing sets (80/20)

2. **Modeling**
   - A `RandomForestClassifier` is trained using scikit-learn
   - Probabilities for fraud (`fraud_probability`) are generated using `predict_proba`

3. **Evaluation**
   - Model performance is evaluated with:
     - Precision, Recall, F1-score
     - Confusion matrix
     - ROC-AUC score

4. **Output**
   - A new dataset is generated containing:
     - All features
     - Actual and predicted fraud labels
     - Fraud probability
     - Original identifiers (`nameOrig`, `nameDest`)
   - Final output is saved as `fraud_predictions.csv`

5. **Visualization**
   - Histogram of non-zero fraud probabilities
   - Trendline of fraud probability scores
   - Top 20 most suspicious transactions
   - Precision-Recall vs Threshold curve

---

## Key Files

- `fraud_predictions.csv` – Output file with predictions
- `Synthetic_Financial_datasets_log.csv` – Input dataset (not included here)
- `fraud_detection_model.pkl` – Saved Random Forest model (optional future addition)

---

## How to Use

1. Upload the dataset to your working directory or Google Colab
2. Run the notebook/script to preprocess the data, train the model, and generate predictions
3. Review visualizations and `fraud_predictions.csv` for potential fraud cases

---

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

You can install dependencies with:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
