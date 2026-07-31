# 💼 Employee Salary Prediction System

A Machine Learning web application built with Python and Streamlit that predicts whether an employee earns more than $50K per year based on demographic and employment information.

## 🚀 Features

- Employee Salary Prediction using Logistic Regression
- Interactive Streamlit Web Application
- Real-time Salary Prediction
- Automatic Feature Encoding
- Feature Scaling using StandardScaler
- Clean and Professional User Interface

## 📂 Project Structure

```
Employee_Salary_Predictor/

├── Dataset/
│   └── adult.csv
│
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── feature.pkl
│
├── App.py
├── Training.py
├── requirements.txt
└── README.md
```

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## 📊 Machine Learning Workflow

1. Load Adult Income Dataset
2. Data Cleaning
3. Feature Encoding
4. Feature Scaling
5. Train Logistic Regression Model
6. Save Trained Model Files
7. Predict Employee Salary using Streamlit

## ▶️ Run The Project

Install all required libraries

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run App.py
```

If the above command doesn't work, use:

```bash
python -m streamlit run App.py
```

## 👨‍💻 Developed By

**Habeeb Khan**
