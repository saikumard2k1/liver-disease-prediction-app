# Liver Disease Screening: Prediction from Routine Blood Tests

An end-to-end machine learning project that predicts whether a patient is likely to have liver disease from ten routine clinical values, deployed as an interactive **Streamlit** web app.

**Live app:** `<add your Render URL here after deployment>`
*(Hosted on Render's free tier. If the app has been idle, the first load can take 30-60 seconds while it wakes up.)*

> **Disclaimer:** This is a screening aid built for learning and demonstration. It is not a medical diagnosis and must not replace the judgement of a qualified clinician.

---

## 1. Problem Statement

Liver disease is increasingly common, driven by alcohol consumption, contaminated food, harmful gases and drugs. Early screening can reduce the burden on doctors, but the biochemical markers of healthy and diseased patients overlap heavily.

**Goal:** build and compare several classifiers on the Indian Liver Patient Dataset, choose a model for production, and deploy it so that a user can enter blood-test values (or upload a CSV) and get a risk assessment.

The project covers:
1. A complete data analysis report (EDA).
2. Predictive models with several classifiers.
3. An analysis of the basis on which the final model was chosen.
4. A model comparison report, a report on challenges faced, and a deployed web app.

## 2. Dataset

- **Source:** Indian Liver Patient Dataset (ILPD), collected in the north-east of Andhra Pradesh, India.
- **Size:** 583 records (416 liver patients, 167 non-liver patients; 441 male, 142 female). After removing 13 duplicates: **570 records**, about 71% with liver disease and 29% without.
- **Features (10):** Age, Gender, Total Bilirubin, Direct Bilirubin, Alkaline Phosphotase, Alamine Aminotransferase (ALT), Aspartate Aminotransferase (AST), Total Proteins, Albumin, Albumin and Globulin Ratio.
- **Target:** 1 = liver disease, 2 = no liver disease in the raw data (recoded to 1 / 0 in the notebook).
- Only 4 missing values, all in Albumin and Globulin Ratio. Ages above 89 are recorded as 90.

## 3. Approach

### Exploratory data analysis
- Univariate, bivariate and multivariate analysis with histograms, count plots, box plots, a pair plot and a correlation heat map.
- Bilirubin, ALP, ALT and AST are heavily right-skewed with many clinically meaningful outliers. Disease patients show higher bilirubin and enzyme values and lower albumin; Total Proteins barely differs between the classes.
- The two classes overlap strongly, so no single feature separates them. This limits the achievable accuracy.
- Strongly correlated pairs: Total vs Direct Bilirubin, ALT vs AST, Total Proteins vs Albumin.

### Preprocessing and feature engineering
| Step | Technique | Reason |
|---|---|---|
| Duplicates (13) | Removed | The same patient must not appear in both train and test sets |
| Missing values (4) | Median imputation | Robust to skew and outliers |
| Encoding | Gender: Male = 1, Female = 0; Target: 1 = disease, 0 = no disease | Numeric inputs for the models |
| Skewed features | `log1p` on the five bilirubin / enzyme features | Reduces the influence of extreme values (skew 3.2-10.6 down to 1.3-1.7) |
| Outliers | Kept, not removed | Very high values indicate severe disease and are the most informative cases |
| Multicollinearity | All features kept | Each is clinically meaningful; tree models tolerate correlation and the final model is regularised |
| Split | Stratified 80/20 (456 train / 114 test) | Preserves the class ratio |
| Scaling | `StandardScaler` fitted on training data only | Avoids test-set leakage |
| Class imbalance | `SMOTENC` on the training set only (Gender as categorical) | Balances classes without generating invalid Gender values and without touching the test set |

### Modelling
Seven classifiers were trained as baselines and then tuned: Logistic Regression, SVM, KNN, Decision Tree, Random Forest, Gradient Boosting and XGBoost.

**Tuning:** `GridSearchCV` with 5-fold shuffled stratified cross-validation, scored on ROC-AUC. SMOTENC sits **inside an `imblearn` Pipeline**, so synthetic samples are generated only from the training part of each fold.

> An earlier version applied SMOTE before cross-validation. That inflated CV scores (Gradient Boosting reached 0.89 ROC-AUC) because validation folds contained synthetic patients built from training rows. The corrected pipeline gives about 0.70 for the same model.

## 4. Results

### Model comparison (test set, default threshold 0.5)

| Model | CV ROC-AUC | Test ROC-AUC | Accuracy | Precision | Recall | Missed patients | False alarms |
|---|---|---|---|---|---|---|---|
| SVM | 0.719 | 0.840 | 0.711 | 0.929 | 0.642 | 29 | 4 |
| **Logistic Regression** | **0.723** | 0.828 | 0.719 | 0.915 | 0.667 | 27 | 5 |
| KNN | 0.679 | 0.827 | 0.746 | 0.906 | 0.716 | 23 | 6 |
| Random Forest | 0.717 | 0.812 | 0.737 | 0.849 | 0.765 | 19 | 11 |
| Gradient Boosting | 0.697 | 0.789 | 0.719 | 0.845 | 0.741 | 21 | 11 |
| XGBoost | 0.698 | 0.785 | 0.746 | 0.851 | 0.778 | 18 | 11 |
| Decision Tree | 0.699 | 0.757 | 0.702 | 0.851 | 0.704 | 24 | 10 |

(The test set has 114 patients: 81 with disease and 33 without.)

**Key findings**
- Simple, regularised models performed best. Every model preferred conservative settings, because the dataset is small and the classes overlap.
- Boosting and bagging models did not beat Logistic Regression: they catch more patients at the default cut-off only by flagging more people, and their ROC-AUC is lower.
- Accuracy (0.70-0.75) is close to the 0.711 obtained by predicting "disease" for everyone, so precision, recall and ROC-AUC are the more informative metrics.

### Final model: Logistic Regression with a tuned decision threshold

At the default threshold of 0.5, Logistic Regression missed 27 of 81 diseased patients (33%), which is too many for screening, where a missed patient costs more than an extra test.

The **decision threshold** is the cut-off that turns the model's score into "disease" or "no disease". It was lowered to **0.46**, chosen from out-of-fold predictions on the training data (so the test set was not used) to catch at least 85% of diseased patients.

| Metric (test set) | Threshold 0.50 | Threshold 0.46 |
|---|---|---|
| Missed diseased patients | 27 of 81 | **13 of 81** |
| False alarms | 5 of 33 | 16 of 33 |
| Recall | 0.667 | **0.840** |
| Precision | 0.915 | 0.810 |
| Accuracy | 0.719 | 0.746 |
| F1 | 0.771 | 0.824 |
| ROC-AUC | 0.828 | 0.828 |

ROC-AUC is unchanged because it measures how well the model *ranks* patients; the threshold only moves the operating point along the ROC curve. The price of higher recall is more follow-up tests for healthy patients, which is unavoidable given the class overlap.

### Feature importance (permutation importance, Logistic Regression)
Alkaline Phosphotase, Direct Bilirubin, Total Bilirubin and ALT are the most influential features, followed by Age, Gender and Albumin. AST and Total Proteins contribute almost nothing. Correlated features share credit, so individual ranks are not stable, but bilirubin and liver-enzyme markers consistently matter more than demographics, which agrees with clinical knowledge.

## 5. Web App

The Streamlit app loads the saved model, scaler and threshold and applies exactly the same preprocessing as the notebook.

- **Single patient tab:** enter the ten values and get a risk score and a lower / higher risk message. The form checks that direct bilirubin does not exceed total bilirubin.
- **Batch / evaluate tab:** upload a CSV; the app returns predictions and, if a `Target` column is present, reports accuracy, precision, recall, F1, ROC-AUC and a confusion matrix. Uploading `sample_test_patients.csv` reproduces the notebook's test results.
- **Threshold slider:** starts at the saved threshold (0.46) and can be adjusted to trade missed patients against false alarms.
- The output is a **risk score, not a calibrated probability** (balanced training data and strong regularisation compress scores to about 0.41-0.76). Compare the score with the threshold, not with 50%.

## 6. Project Structure

```
liver_app/
├── app.py                                  # Streamlit interface
├── preprocessing.py                        # Preprocessing shared with the notebook
├── liver_model.pkl                         # Trained model, scaler and threshold
├── sample_test_patients.csv                # Test patients for the batch tab
├── Liver_Patients_Disease_Prediction.ipynb # Full analysis: EDA, modelling, tuning, evaluation
├── Data/                                   # Raw dataset
├── requirements.txt                        # Pinned dependencies
├── render.yaml                             # Render deployment settings
└── .streamlit/config.toml                  # Streamlit server settings
```

## 7. Run Locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. The model was built with Python 3.13.9, scikit-learn 1.9.1, pandas 2.3.3 and numpy 2.3.5; the pinned versions in `requirements.txt` are needed to load the saved model.

## 8. Deployment

The app is deployed as a Render web service:

- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- Environment variable: `PYTHON_VERSION = 3.13.9`

## 9. Limitations and Future Work

- Small, single-region dataset; only 33 healthy patients in the test set, so test metrics are uncertain (ROC-AUC carries roughly ±0.08). Cross-validated AUC (about 0.72) is the more cautious estimate of real-world performance.
- Scores are not calibrated probabilities.
- Future work: larger multi-centre data with external validation, probability calibration, additional clinical features (for example the AST/ALT ratio), and a pipeline object that bundles preprocessing and the model.

## 10. Tech Stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, Matplotlib, Seaborn, Streamlit, joblib, Render, GitHub.

## 11. Author

`Sai Kumar Dandupati` | `https://github.com/saikumard2k1`

Capstone project PRCP-1007 (Liver Patient Disease Prediction), Datamites.
