# 🧠 Breast Cancer Prediction System

An end-to-end machine learning application that predicts whether a tumor is **malignant (cancerous)** or **benign (non-cancerous)** using diagnostic data.

---

## 🚀 Live Demo
🌐 https://breast-cancer-predictor-koushik.streamlit.app

---

## 📊 Dataset
- Breast Cancer Wisconsin Dataset (UCI Machine Learning Repository)

---

## ⚙️ Features Used
- radius_worst  
- perimeter_worst  
- area_worst  
- concave points_worst  
- radius_mean  
- texture_mean  
- concavity_mean  

---

## 🛠️ Feature Engineering
- Removed irrelevant columns (e.g., ID, empty fields)  
- Selected a subset of important features to balance model performance and usability  
- Applied feature scaling using StandardScaler  

---

## 📊 Model Experimentation
Detailed model comparison and experimentation can be found in:
👉 **model_comparison.ipynb**

This includes evaluation of multiple models such as Random Forest, XGBoost, and Gradient Boosting.

---

## 🤖 Models Evaluated
- Random Forest (selected model)  
- XGBoost  
- Gradient Boosting  

👉 Random Forest was selected due to its consistent performance and interpretability.

---

## 📈 Performance
- Accuracy: ~95%  
- Strong recall for malignant cases  

---

## 🧩 Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  

---

## 💻 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
