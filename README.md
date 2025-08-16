# Machine Learning Projects

This repository contains two main machine learning projects:

## 1. Adult Census Income Prediction

**Objective:**
Predict whether an individual's income exceeds $50K/year based on census data.

**Dataset:**
- UCI Adult Census dataset (CSV format)
- Features include age, workclass, education, marital status, occupation, race, sex, capital gain/loss, hours per week, native country, and salary class.

**Process:**
- Data cleaning and preprocessing (handling missing values, encoding categorical variables, feature engineering)
- Exploratory Data Analysis (EDA): Univariate and bivariate analysis of features
- Model building: Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, Neural Network
- Model evaluation: Accuracy, F1-score, confusion matrix, classification report
- Ensemble modeling for improved performance

**Results:**
- Multiple models compared; ensemble model achieves the best accuracy and F1-score on both train and test sets.

## 2. Churn Prediction

**Objective:**
Predict customer churn for a financial institution using customer transaction and demographic data.

**Dataset:**
- Custom churn dataset (`churn_prediction.csv`)
- Features include customer demographics, account balances, transaction history, dependents, occupation, city, branch code, etc.

**Process:**
- Data cleaning (handling missing values, outlier treatment, feature transformation)
- Feature engineering and encoding (label and one-hot encoding)
- Exploratory Data Analysis (EDA): Univariate and bivariate analysis
- Model building: Logistic Regression, KNN, Decision Tree, Random Forest
- Model evaluation: Accuracy, F1-score
- Ensemble modeling for final predictions

**Results:**
- Ensemble model provides robust predictions and improved accuracy compared to individual models.

---

## How to Run

1. Open the Jupyter notebooks (`Adult_census_income.ipynb` and `churn_prediction/churn_prediction.ipynb`) for step-by-step code and analysis.
2. Ensure required Python packages are installed: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow` (for neural network).
3. Place the datasets in the correct paths as referenced in the notebooks.

## Project Structure
- `Adult_census_income.ipynb`: Notebook for census income prediction
- `churn_prediction/churn_prediction.ipynb`: Notebook for churn prediction
- `churn_prediction/churn_prediction.csv`: Dataset for churn prediction
- `README.md`: Project overview and instructions

---

For details, see the notebooks for code, analysis, and results.
