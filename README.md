# 🍷 Wine Quality Prediction App

A **Streamlit-based Machine Learning application** that predicts wine quality based on physicochemical properties.  
Built using the **Wine Quality Dataset** from the UCI Machine Learning Repository, this project demonstrates the complete ML workflow — from data preprocessing to cloud deployment.

---

## 📊 Dataset Description

- **Source:** [UCI Machine Learning Repository – Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Shape:** `1599 rows × 12 columns` (red wine data)
- **Features:**
  - Fixed acidity
  - Volatile acidity
  - Citric acid
  - Residual sugar
  - Chlorides
  - Free sulfur dioxide
  - Total sulfur dioxide
  - Density
  - pH
  - Sulphates
  - Alcohol
- **Target:** Wine quality score (0–10)

**Rationale for Selection:**  
The dataset offers real-world tabular data with balanced features and a clear predictive target, making it ideal for demonstrating an **end-to-end ML workflow**.

---

## 🛠 Data Preprocessing

- Checked dataset **shape** and **structure**.
- Verified **no missing values**.
- Removed **duplicate rows** to ensure data integrity.
- Standardized feature scales for improved model performance.
- Split into **training** and **testing sets** for evaluation.

---

## 🤖 Model Selection & Evaluation

- **Model Used:** Random Forest Classifier (selected for its accuracy and robustness with tabular data).
- **Hyperparameters:** Tuned using **GridSearchCV** to optimize tree depth, number of trees, and split criteria.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score.
- **Best Accuracy Achieved:** ~90% on the test set.

---

## 🎨 Streamlit App Design

- **Sidebar:**
  - Sliders and input boxes for each physicochemical property.
- **Main Page:**
  - Displays **predicted wine quality**.
  - Shows **model performance metrics**.
  - Includes **sample dataset preview**.
- **Design Goals:**
  - Simple and intuitive user interface.
  - Immediate prediction feedback.
  - Accessible to both technical and non-technical users.

**Live App:** [Wine Quality Prediction App](https://thulanithisarani-wine-quality-app-75fztt.streamlit.app/)

---

## 🚀 Deployment

- **Platform:** Streamlit Cloud  
- **Repository:** Connected directly to [GitHub Project](https://github.com/ThulaniThisarani/wine_quality)  
- **Requirements:** Dependencies managed in `requirements.txt` for version compatibility.  

---

## ⚠️ Challenges Faced

- **Dependency version conflicts** (`scikit-learn` and `pandas`).
- **Model file size optimization** for faster loading.
- **Resource constraints** in Streamlit Cloud (RAM, CPU).

---

## 🖼 Screenshots

*(Add actual screenshots here)*  

---

## 📚 Learning Outcomes

- Gained full ML pipeline experience: **preprocessing → training → evaluation → deployment**.
- Improved skills in **model tuning** and **UI design**.
- Learned **deployment best practices** and **dependency management**.
- **Future plans:**
  - Add SHAP-based model interpretability.
  - Include interactive visualizations for better user engagement.

---

## 📦 Installation & Local Run

```bash
# Clone repository
git clone https://github.com/ThulaniThisarani/wine_quality.git
cd wine_quality

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
