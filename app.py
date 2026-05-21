import pickle
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

# ── Load model (uses pickle, matching the notebook exactly) ──
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ── App ──────────────────────────────────────────────────────
app = FastAPI(
    title="Adult Income Classifier API",
    description=(
        "Predicts whether a person earns >50K or <=50K per year. "
        "Pipeline: SimpleImputer → StandardScaler / OneHotEncoder → Decision Tree. "
        "Trained on UCI Adult Census dataset (Accuracy: 86.08%)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schema — mirrors notebook's X exactly ────────────
class PredictRequest(BaseModel):
    age: int = Field(..., ge=17, le=90, example=35)
    workclass: Literal[
        "Private", "Self-emp-not-inc", "Self-emp-inc",
        "Federal-gov", "Local-gov", "State-gov",
        "Without-pay", "Never-worked"
    ] = Field(..., example="Private")
    fnlwgt: int = Field(..., ge=0, example=189778)
    education: Literal[
        "Bachelors", "Some-college", "11th", "HS-grad",
        "Prof-school", "Assoc-acdm", "Assoc-voc", "9th",
        "7th-8th", "12th", "Masters", "1st-4th", "10th",
        "Doctorate", "5th-6th", "Preschool"
    ] = Field(..., example="Bachelors")
    educational_num: int = Field(..., alias="educational-num", ge=1, le=16, example=13)
    marital_status: Literal[
        "Married-civ-spouse", "Divorced", "Never-married",
        "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ] = Field(..., alias="marital-status", example="Married-civ-spouse")
    occupation: Literal[
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
    ] = Field(..., example="Exec-managerial")
    relationship: Literal[
        "Wife", "Own-child", "Husband",
        "Not-in-family", "Other-relative", "Unmarried"
    ] = Field(..., example="Husband")
    race: Literal[
        "White", "Asian-Pac-Islander",
        "Amer-Indian-Eskimo", "Other", "Black"
    ] = Field(..., example="White")
    gender: Literal["Male", "Female"] = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", ge=0, example=0)
    capital_loss: int = Field(..., alias="capital-loss", ge=0, example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", ge=1, le=99, example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")

    model_config = {"populate_by_name": True}


# ── Response schema ───────────────────────────────────────────
class PredictResponse(BaseModel):
    prediction: str
    prediction_code: int
    confidence_note: str


# ── Routes ────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "message": "Adult Income Classifier API is running 🚀",
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "model": "Decision Tree",
        "encoder": "OneHotEncoder",
        "accuracy": "86.08%",
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(data: PredictRequest):
    # Reconstruct DataFrame with exact column names the pipeline expects
    input_dict = {
        "age":             data.age,
        "workclass":       data.workclass,
        "fnlwgt":          data.fnlwgt,
        "education":       data.education,
        "educational-num": data.educational_num,
        "marital-status":  data.marital_status,
        "occupation":      data.occupation,
        "relationship":    data.relationship,
        "race":            data.race,
        "gender":          data.gender,
        "capital-gain":    data.capital_gain,
        "capital-loss":    data.capital_loss,
        "hours-per-week":  data.hours_per_week,
        "native-country":  data.native_country,
    }
    df = pd.DataFrame([input_dict])
    pred  = int(model.predict(df)[0])
    label = ">50K" if pred == 1 else "<=50K"

    return PredictResponse(
        prediction=label,
        prediction_code=pred,
        confidence_note="Predicted by Decision Tree (OneHotEncoder pipeline, Accuracy: 86.08%)"
    )


@app.get("/model-info", tags=["Model"])
def model_info():
    return {
        "model_type":      "Decision Tree Classifier",
        "accuracy":        "86.08%",
        "dataset":         "UCI Adult Census Income",
        "total_features":  14,
        "numeric_features": ["age", "fnlwgt", "educational-num",
                              "capital-gain", "capital-loss", "hours-per-week"],
        "categorical_features": ["workclass", "education", "marital-status",
                                  "occupation", "relationship", "race",
                                  "gender", "native-country"],
        "pipeline": [
            "SimpleImputer (median for numeric, most_frequent for categorical)",
            "StandardScaler (numeric)",
            "OneHotEncoder handle_unknown=ignore (categorical)",
            "DecisionTreeClassifier max_depth=10"
        ],
        "classes": ["<=50K", ">50K"],
    }
