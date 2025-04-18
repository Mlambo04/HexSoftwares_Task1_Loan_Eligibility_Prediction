# 🏦 Loan Eligibility Prediction — HexSoftwares Internship Project

### 🔍 Project Description
This project is part of my internship with **@HexSoftwares**, where the goal is to build a **Loan Eligibility Prediction System**. The model predicts whether a person will be approved for a loan based on their income, education, credit history, and other factors.

---

### 📁 Dataset Overview
The dataset includes the following fields:
- **Gender**
- **Married**
- **Dependents**
- **Education**
- **Self_Employed**
- **ApplicantIncome**
- **CoapplicantIncome**
- **LoanAmount**
- **Loan_Amount_Term**
- **Credit_History**
- **Property_Area**
- **Loan_Status** (Target: Approved/Rejected)

---

### 🧠 What I Did
1. **Explored the dataset** to understand patterns and missing data.
2. **Cleaned the data** and handled missing values.
3. **Engineered features** (like Total_Income, EMI, etc.)
4. **Trained ML models** (Random Forest Classifier gave the best result)
5. **Built a Flask API** to accept loan application inputs and return predictions.
6. **Deployed locally** using Flask for demonstration.

---

### ⚙️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Flask
- Swagger / Postman for API testing

---

### 🖥️ Run it Locally
```bash
# Step 1: Clone the repo
git clone https://github.com/YourUsername/HexSoftwares_Loan_Eligibility_Prediction

# Step 2: Navigate to the folder
cd HexSoftwares_Loan_Eligibility_Prediction

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the Flask app
python app.py

