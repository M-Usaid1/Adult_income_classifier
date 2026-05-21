import streamlit as st
import requests
import json

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Income Classifier",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* ── Header ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #6b7280;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
}

/* ── Accuracy badges ── */
.badge-row {
    display: flex;
    justify-content: center;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 1.2rem 0 2rem 0;
}
.badge {
    background: #13131a;
    border: 1px solid #2a2a3a;
    border-radius: 999px;
    padding: 0.4rem 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #9ca3af;
}
.badge span { color: #a78bfa; font-weight: 500; }
.badge.best { border-color: #a78bfa; color: #c4b5fd; }
.badge.best span { color: #34d399; }

/* ── Section label ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: #4b5563;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    margin-top: 1.5rem;
}

/* ── Card ── */
.card {
    background: #13131a;
    border: 1px solid #1e1e2e;
    border-radius: 16px;
    padding: 1.6rem;
    margin-bottom: 1rem;
}

/* ── Result box ── */
.result-high {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #34d399;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-low {
    background: linear-gradient(135deg, #1e1b4b, #1e1e3f);
    border: 1px solid #818cf8;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-label {
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 0.5rem;
}
.result-value {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
}
.result-high .result-value { color: #34d399; }
.result-low  .result-value { color: #a5b4fc; }
.result-note {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    opacity: 0.6;
    margin-top: 0.6rem;
}

/* ── Streamlit widget overrides ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"]  > div {
    background: #0d0d14 !important;
    border-color: #2a2a3a !important;
    border-radius: 10px !important;
    color: #e8e8f0 !important;
    font-family: 'DM Mono', monospace !important;
}
.stSlider > div > div > div { background: #a78bfa !important; }
label { color: #9ca3af !important; font-size: 0.85rem !important; }

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    margin-top: 1rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Divider ── */
hr { border-color: #1e1e2e !important; margin: 1.5rem 0 !important; }

/* ── Hide default elements ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>💰 Income Classifier</h1>
    <p>UCI Adult Census · Decision Tree · Binary Classification</p>
</div>
<div class="badge-row">
    <div class="badge">Logistic Regression <span>85.24%</span></div>
    <div class="badge best">Decision Tree <span>86.08%</span> ★ best</div>
    <div class="badge">SVC <span>85.96%</span></div>
    <div class="badge">KNN <span>83.37%</span></div>
</div>
""", unsafe_allow_html=True)

# ── API URL ───────────────────────────────────────────────────
API_URL = "http://localhost:8000/predict"

# ── Layout ───────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1], gap="large")

# ── Column 1 — Personal Info ──────────────────────────────────
with col1:
    st.markdown('<div class="section-label">👤 Personal Info</div>', unsafe_allow_html=True)
    age = st.slider("Age", 17, 90, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    race = st.selectbox("Race", [
        "White", "Black", "Asian-Pac-Islander",
        "Amer-Indian-Eskimo", "Other"
    ])
    native_country = st.selectbox("Native Country", [
        "United-States", "Mexico", "Philippines", "Germany",
        "Canada", "Puerto-Rico", "El-Salvador", "India",
        "Cuba", "England", "Jamaica", "South", "China",
        "Italy", "Dominican-Republic", "Vietnam", "Guatemala",
        "Japan", "Columbia", "Poland", "Other"
    ])
    marital_status = st.selectbox("Marital Status", [
        "Married-civ-spouse", "Never-married", "Divorced",
        "Separated", "Widowed",
        "Married-spouse-absent", "Married-AF-spouse"
    ])
    relationship = st.selectbox("Relationship", [
        "Husband", "Wife", "Own-child",
        "Not-in-family", "Unmarried", "Other-relative"
    ])

# ── Column 2 — Work & Education ───────────────────────────────
with col2:
    st.markdown('<div class="section-label">🎓 Education & Work</div>', unsafe_allow_html=True)
    education = st.selectbox("Education", [
        "Bachelors", "Some-college", "HS-grad", "Masters",
        "Doctorate", "Prof-school", "Assoc-acdm", "Assoc-voc",
        "11th", "10th", "9th", "12th", "7th-8th",
        "1st-4th", "5th-6th", "Preschool"
    ])
    educational_num = st.slider("Education Level (1–16)", 1, 16, 13,
        help="1=Preschool → 16=Doctorate")
    workclass = st.selectbox("Work Class", [
        "Private", "Self-emp-not-inc", "Self-emp-inc",
        "Federal-gov", "Local-gov", "State-gov",
        "Without-pay", "Never-worked"
    ])
    occupation = st.selectbox("Occupation", [
        "Exec-managerial", "Prof-specialty", "Tech-support",
        "Craft-repair", "Sales", "Adm-clerical",
        "Transport-moving", "Handlers-cleaners",
        "Machine-op-inspct", "Farming-fishing",
        "Other-service", "Protective-serv",
        "Priv-house-serv", "Armed-Forces"
    ])
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    fnlwgt = st.number_input("FNLWGT (Census Weight)", min_value=0,
                              value=189778, step=1000,
                              help="Sampling weight assigned by Census Bureau")

# ── Column 3 — Financial + Predict ───────────────────────────
with col3:
    st.markdown('<div class="section-label">💵 Financial Details</div>', unsafe_allow_html=True)
    capital_gain = st.number_input("Capital Gain ($)", min_value=0,
                                    max_value=99999, value=0, step=500)
    capital_loss = st.number_input("Capital Loss ($)", min_value=0,
                                    max_value=99999, value=0, step=500)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">🔍 Prediction</div>', unsafe_allow_html=True)

    predict_btn = st.button("⚡ Predict Income", use_container_width=True)

    if predict_btn:
        payload = {
            "age":             age,
            "workclass":       workclass,
            "fnlwgt":          fnlwgt,
            "education":       education,
            "educational-num": educational_num,
            "marital-status":  marital_status,
            "occupation":      occupation,
            "relationship":    relationship,
            "race":            race,
            "gender":          gender,
            "capital-gain":    capital_gain,
            "capital-loss":    capital_loss,
            "hours-per-week":  hours_per_week,
            "native-country":  native_country,
        }
        try:
            with st.spinner("Calling model..."):
                response = requests.post(API_URL, json=payload, timeout=10)

            if response.status_code == 200:
                result = response.json()
                pred   = result["prediction"]
                if pred == ">50K":
                    st.markdown(f"""
                    <div class="result-high">
                        <div class="result-label">Predicted Annual Income</div>
                        <div class="result-value">{'> $50K'}</div>
                        <div class="result-note">High income earner 🚀</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-low">
                        <div class="result-label">Predicted Annual Income</div>
                        <div class="result-value">{'≤ $50K'}</div>
                        <div class="result-note">Standard income range 📊</div>
                    </div>""", unsafe_allow_html=True)

                with st.expander("🔎 Raw API response"):
                    st.json(result)
            else:
                st.error(f"API Error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot reach API. Make sure `uvicorn app:app --reload` is running on port 8000.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

    else:
        st.markdown("""
        <div style="
            background:#13131a;
            border:1px dashed #2a2a3a;
            border-radius:16px;
            padding:2rem;
            text-align:center;
            margin-top:0.5rem;
            color:#374151;
            font-family:'DM Mono',monospace;
            font-size:0.8rem;
        ">
            Fill in the fields<br>and hit Predict ↑
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; font-family:'DM Mono',monospace;
            font-size:0.72rem; color:#374151; padding-bottom:1rem;">
    Adult Income Classifier · Decision Tree · UCI Census Dataset · FastAPI + Streamlit
</div>
""", unsafe_allow_html=True)
