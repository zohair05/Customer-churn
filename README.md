# 🔄 Telecom Customer Churn Predictor
An end-to-end Machine Learning web application built with Python and Streamlit. This project predicts whether a telecom customer is likely to cancel their subscription (churn) and uses **SHAP (Shapley Additive Explanations)** to provide transparency by explaining *why* the model made its decision.

---

## 🌟 Features

* **Data Preprocessing**
  Handles missing values and performs encoding of categorical variables.

* **Robust Modeling**
  Uses a Random Forest Classifier with hyperparameter tuning via `GridSearchCV`.

* **Explainable AI (XAI)**
  Integrates SHAP to show how each feature (e.g., tenure, contract type, monthly charges) influences predictions.

* **Interactive UI**
  Clean and simple interface built using Streamlit.

---

## 📂 Project Structure

```
customer_churn_project/
│
├── dataset/
│   └── Telco-Customer-Churn.csv
├── model/
│   ├── churn_model.pkl
│   └── model_columns.pkl
├── output/
│   ├── correlation_heatmap.png
│   └── confusion_matrix.png
├── app.py
├── model.py
├── preprocess.py
└── requirements.txt
```

---

## ⚙️ Setup Guide to run the project locally.


### 1. Prepare the Dataset

1. Download the **Telco Customer Churn dataset** from Kaggle.
2. Create a folder named `dataset` inside your project directory.
3. Place the CSV file inside the folder and rename it:
```
Telco-Customer-Churn.csv
```

---

### 2. Create Virtual Environment

Open terminal inside your project folder and run:

```bash
python -m venv venv
```

Activate environment:

```bash
.\venv\Scripts\activate
```
---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Train the Model (Run Once)

```bash
python model.py
```

This step will:

* Load dataset
* Perform EDA
* Train model
* Save `.pkl` files

You only need to run this once.

---

### 5. Run the Application

```bash
python -m streamlit run app.py
```

App will open in browser at:

```
http://localhost:8501
```

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* Streamlit
* Pandas & NumPy
* SHAP
* Matplotlib & Seaborn

---

## 📊 Output

* Customer churn prediction (Yes / No)
* Model evaluation metrics
* SHAP-based feature explanations
* Interactive dashboard

---
