# Wine Dataset Classification Project

## Project Overview
This project focuses on **classifying wines into three categories** using the Wine dataset from `sklearn.datasets`. The workflow includes **data exploration, visualization, preprocessing, model training, evaluation, and comparison of multiple classification algorithms**.

---

## Dataset
- **Source:** `sklearn.datasets.load_wine()`
- **Rows:** 178
- **Columns:** 13 numeric features (e.g., alcohol, malic_acid, flavanoids, proline)
- **Target:** Wine class (0, 1, 2)

---

## Workflow & Steps

### 1. Data Exploration
- Loaded dataset into **pandas DataFrame**.
- Checked **data types**, **missing values**, and **basic statistics**.
- Visualized feature correlations using **heatmaps**.
- Examined feature distributions using **histograms**.
- Checked class distribution using **bar plots**.

### 2. Data Preparation
- Split dataset into **training (80%)** and **test (20%)** sets using `train_test_split`.
- Ensured no missing values existed in features or target.

### 3. Model Training & Evaluation
- Trained multiple classification models:
  - **Logistic Regression**
  - **Decision Tree**
  - **Random Forest**
  - **K-Nearest Neighbors (KNN)**
  - **Naive Bayes**
  - **Support Vector Machine (SVM)**
- Evaluated models using:
  - **Accuracy**
  - **Precision, Recall, F1-score** (`classification_report`)
  - **Confusion matrices** (visualized using `seaborn.heatmap`)

### 4. Results
| Model                   | Accuracy |
|-------------------------|----------|
| Logistic Regression     | 1.00     |
| Random Forest           | 1.00     |
| Naive Bayes             | 1.00     |
| Decision Tree           | 0.94     |
| Support Vector Machine  | 0.81     |
| K-Nearest Neighbors     | 0.72     |

- **Best performing models:** Logistic Regression, Random Forest, Naive Bayes (100% accuracy)
- Visualized model performance using **bar chart** for comparison.

---

## Skills Demonstrated
- **Data Analysis & Exploration:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn (heatmaps, histograms, bar plots, confusion matrices)
- **Data Preprocessing:** Train-test split, handling numeric features, class analysis
- **Machine Learning Models:** Logistic Regression, Decision Trees, Random Forest, KNN, Naive Bayes, SVM
- **Model Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Comparative Analysis:** Evaluating multiple models to select the best-performing algorithm

---

## Outputs
- **Confusion matrices** for all models
- **Classification reports** showing precision, recall, F1-score
- **Accuracy comparison chart** for visualizing model performance
- **Trained models** ready for predicting new wine samples

---

