# 💰 Adult Income Classifier

A complete end-to-end Machine Learning classification project that predicts whether a person earns **>50K or ≤50K per year** based on census data.

Built with **scikit-learn**, served via **FastAPI**, and visualized through a **Streamlit** frontend.

---

## 📌 Project Overview

This project uses the [UCI Adult Census Income dataset](https://archive.ics.uci.edu/ml/datasets/adult) (48,842 records, 14 features) to train and compare four classification models. The best-performing model is saved and deployed as a REST API with an interactive web UI.

**Target variable:** `income` → binary (`>50K` = 1, `<=50K` = 0)

---

## 📊 Model Accuracies

| Model | Accuracy |
|---|---|
| Logistic Regression | 85.24% |
| **Decision Tree** | **86.08% 🏆** |
| SVC | 85.96% |
| KNN | 83.37% |

> Train/Test split: 80/20 · Stratified · Random state: 42

---

## 🏆 Best Model — Decision Tree

| Property | Detail |
|---|---|
| Algorithm | DecisionTreeClassifier |
| Max Depth | 10 |
| Encoder | OneHotEncoder (handle_unknown=ignore) |
| Scaler | StandardScaler |
| Imputer | SimpleImputer (median / most_frequent) |
| Test Accuracy | **86.08%** |
| Saved as | model.pkl |

The Decision Tree outperforms the other models on this dataset because the income prediction problem is fundamentally rule-based — combinations of occupation, education, and marital status map naturally to tree splits.

---

## 📁 Project Structure

```
adult-income-classifier/
├── Adult_dataset.ipynb     # EDA + model training + comparison
├── model.pkl               # Saved best model (Decision Tree pipeline)
├── app.py                  # FastAPI backend
├── streamlit_app.py        # Streamlit frontend
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚙️ Steps to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/your-username/adult-income-classifier.git
cd adult-income-classifier
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI backend
```bash
uvicorn app:app --reload
```
- API → `http://localhost:8000`
- Swagger docs → `http://localhost:8000/docs`

### 5. Run the Streamlit frontend
Open a **second terminal** (venv activated):
```bash
streamlit run streamlit_app.py
```
- App → `http://localhost:8501`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Model status + accuracy |
| POST | `/predict` | Predict income from 14 features |
| GET | `/model-info` | Full pipeline details |

### Example request body for `/predict`
```json
{
  "age": 35,
  "workclass": "Private",
  "fnlwgt": 189778,
  "education": "Bachelors",
  "educational-num": 13,
  "marital-status": "Married-civ-spouse",
  "occupation": "Exec-managerial",
  "relationship": "Husband",
  "race": "White",
  "gender": "Male",
  "capital-gain": 0,
  "capital-loss": 0,
  "hours-per-week": 40,
  "native-country": "United-States"
}
```

### Example response
```json
{
  "prediction": ">50K",
  "prediction_code": 1,
  "confidence_note": "Predicted by Decision Tree (OneHotEncoder pipeline, Accuracy: 86.08%)"
}
```

---

## 🚀 Deployment / Demo Link

> 🔗 **Live Demo:** `https://your-deployment-link-here.com`
> *(Update after deploying to Streamlit Cloud / Render / Railway)*

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML / Data | scikit-learn, pandas, numpy |
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Serialization | pickle |

---

## 👤 Author

**Your Name**
BS Data Science · Your University
GitHub: [@your-username](https://github.com/your-username)
