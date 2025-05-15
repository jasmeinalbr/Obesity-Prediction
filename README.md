# 🧠 Obesity Prediction using Machine Learning

This repository contains a machine learning project to predict obesity categories based on individual demographic and lifestyle features. The model was built using Python and the Scikit-learn library with a dataset from Kaggle.

## 📌 Project Objective

To build and evaluate machine learning models that classify individuals into obesity categories (`Normal weight`, `Overweight`, `Obese`, `Underweight`) based on:

- Age
- Gender
- Height
- Weight
- BMI
- Physical Activity Level

The main goal is to explore patterns in health data and provide a prediction tool that supports early identification of obesity risks.

---

## 📂 Dataset

- **Source**: [Kaggle - Obesity Prediction Dataset](https://www.kaggle.com/datasets/mrsimple07/obesity-prediction)
- **Attributes**:
  - `Age`: Age in years
  - `Gender`: Male / Female
  - `Height`: In cm
  - `Weight`: In kg
  - `BMI`: Body Mass Index
  - `PhysicalActivityLevel`: Ordinal scale from 1 to 4
  - `ObesityCategory`: Target label (e.g., Normal weight, Obese, etc.)

---

## ⚙️ Project Pipeline

### 1. Data Understanding
- Performed EDA (Exploratory Data Analysis) using histograms, KDE plots, and correlation heatmap
- Identified potential outliers in `Weight`, `Height`, and `BMI`

### 2. Data Preparation
- Checked for missing values and duplicates
- Outliers retained (as they may represent valid health conditions)
- Categorical features encoded:
  - `Gender`: One-hot encoding
  - `PhysicalActivityLevel`: Label encoded
  - `ObesityCategory`: Label encoded (used as the target variable)
- Scaled numeric features (`Age`, `Height`, `Weight`, `BMI`) using `StandardScaler`
- Data split into 80% train and 20% test sets

### 3. Modeling
Two models were trained:
- **Logistic Regression** (as baseline)
- **Random Forest Classifier**

### 4. Evaluation
- Metrics used: Accuracy, Precision, Recall, and F1-score (weighted)
- Random Forest showed superior performance:
  ```
  Accuracy: 99.5%
  F1 Score: 0.9950
  ```

- Performed **Hyperparameter Tuning** using `GridSearchCV` on Random Forest:
  ```python
  {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2}
  ```

---

## 📈 Results

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.965    | 0.9658    | 0.965  | 0.9651   |
| Random Forest       | 0.995    | 0.9951    | 0.995  | 0.9950   |

---

## 🧪 Inference Example

A new data sample:
```python
Age = 30
Height = 170 cm
Weight = 70 kg
BMI = 24.2
PhysicalActivityLevel = 3
Gender = Male
```

After preprocessing and prediction, the model outputs:
```
Obesity Category: Normal weight
Confidence: 96.6%
```

---

## 🛠 Tech Stack

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib

---

## 📁 Folder Structure

```
├── assets/                   # Folder for images (EDA, confusion matrix, etc.)
├── data/
│   └── obesity_data.csv
│
├── notebooks/
│   └── obesity_prediction.ipynb
│
├── laporan_project.md
└── README.md
```

---

## 🤝 Acknowledgements

- Dataset by [mrsimple07 on Kaggle](https://www.kaggle.com/datasets/mrsimple07/obesity-prediction)
- WHO & [Tandiono & Sanjaya (2023)](https://doi.org/10.33379/gtech.v8i1.3604) as domain references.