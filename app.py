import joblib
import pandas as pd
import streamlit as st
import json
import xgboost as xgb
st.set_page_config(page_title="Telcom Customer Churn Prediction",
                   page_icon="📉", layout="centered")

THRESHOLD = 0.5   # probability above which a customer is flagged as "likely to churn"

ADDON_SERVICES = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                  'TechSupport', 'StreamingTV', 'StreamingMovies']
CAT_COLS = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod', 'tenure_group']
ALGO_NAMES = {
    "LogisticRegression": "Logistic Regression", "SVC": "SVM",
    "KNeighborsClassifier": "KNN", "DecisionTreeClassifier": "Decision Tree",
    "RandomForestClassifier": "Random Forest", "AdaBoostClassifier": "AdaBoost",
    "XGBClassifier": "XGBoost",
}


def add_features(data):
   
    data = data.copy()
    data['n_services'] = (data[ADDON_SERVICES] == 'Yes').sum(axis=1)
    data['tenure_group'] = pd.cut(data['tenure'], bins=[-1, 12, 24, 48, 72],
                                  labels=['0-12', '13-24', '25-48', '49-72']).astype(str)
    data['avg_charge'] = data['TotalCharges'] / data['tenure'].clip(lower=1)
    return data


@st.cache_resource
def load_artifacts():

    scaler = joblib.load("scaler.pkl")

    model = xgb.XGBClassifier()
    model.load_model("xgb_model.json")

    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)

    return model, scaler, feature_columns


model, scaler, feature_columns = load_artifacts()

final_estimator = model
algo_name = "XGBoost"


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    df = add_features(raw)
    df = pd.get_dummies(df, columns=CAT_COLS)
    
    return df.reindex(columns=feature_columns, fill_value=0).astype(float)



st.markdown("""
<style>
/* page */
.stApp { background: #f3f5f9; }
header[data-testid="stHeader"] { background: #0b1220; }
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 760px; }

/* sidebar */
section[data-testid="stSidebar"] > div { background: linear-gradient(180deg, #0b1a33 0%, #12274a 100%); }
section[data-testid="stSidebar"] * { color: #e8edf6; }
.brand { font-size: 1.05rem; font-weight: 700; letter-spacing: .5px; margin-bottom: 1.6rem; }
.side-title { font-size: 1rem; font-weight: 700; margin: .4rem 0 .6rem 0; }
.side-text { font-size: .85rem; line-height: 1.6; color: #cbd5e6; }
.side-label { font-size: .85rem; font-weight: 700; margin-top: .7rem; }
.side-value { font-size: .85rem; color: #cbd5e6; margin-bottom: .2rem; }
.side-hr { border: none; border-top: 1px solid rgba(255,255,255,.12); margin: 1.6rem 0; }
.side-foot { font-size: .72rem; color: #9fb0cc; margin-top: 2.5rem; }

/* hero banner */
.hero { background: linear-gradient(135deg, #0f2342 0%, #1d4f66 65%, #2a7a78 100%);
        border-radius: 18px; padding: 2.2rem 2.2rem 2rem 2.2rem; margin-bottom: 2.2rem;
        box-shadow: 0 6px 18px rgba(15,35,66,.18); }
.hero h1 { color: #fff; font-size: 1.9rem; margin: 0 0 1.2rem 0; padding: 0; }
.hero p { color: #dbe6f2; font-size: .92rem; line-height: 1.6; margin: 0; }

/* headings */
.page-h2 { font-size: 2rem; font-weight: 600; color: #16213a; margin: 1rem 0 1.4rem 0; }
.section { font-size: 1.15rem; font-weight: 700; color: #16213a; margin: 2.4rem 0 .8rem 0; }

/* inputs: dark selects, light number inputs */
div[data-baseweb="select"] > div { background: #1a1f2b !important; border-radius: 8px; border: none; }
div[data-baseweb="select"] * { color: #fff !important; }
div[data-baseweb="select"] svg { fill: #fff !important; }
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li { background: #1a1f2b !important; color: #fff !important; }
div[data-baseweb="popover"] li:hover { background: #2a3142 !important; }
label p, div[data-testid="stWidgetLabel"] p { color: #4b5568 !important; font-size: .8rem !important; }

/* button */
div.stButton > button { background: #2b7a78; color: #fff; border: none; border-radius: 8px;
                        padding: .6rem 1.4rem; font-weight: 600; margin-top: 1.5rem; }
div.stButton > button:hover { background: #236b69; color: #fff; }

/* result */
.result-card { background: #fff; border-radius: 12px; padding: 2rem 1.5rem; text-align: center;
               box-shadow: 0 2px 10px rgba(15,35,66,.08); border-left: 6px solid #d9534f; }
.result-card.stay { border-left-color: #2e9e6a; }
.result-card h3 { margin: 0 0 .8rem 0; font-size: 1.25rem; font-weight: 600; color: #333; }
.result-card p { margin: 0; color: #4b5568; font-size: .85rem; }
.metric-card { background: #fff; border-radius: 12px; padding: 1.1rem 1.2rem;
               box-shadow: 0 2px 10px rgba(15,35,66,.08); }
.metric-card .lbl { font-size: .75rem; color: #5b6578; }
.metric-card .val { font-size: 1.7rem; font-weight: 700; color: #16213a; margin-top: .2rem; }
.bar-label { font-size: .85rem; font-weight: 700; color: #16213a; margin: 1.2rem 0 .5rem 0; }
.bar-track { background: #1f2430; border-radius: 4px; height: 6px; width: 100%; }
.bar-fill { background: #1f78d4; height: 6px; border-radius: 4px; }
.footer { text-align: center; color: #7a8497; font-size: .75rem; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown(f"""
    <div class="brand">TELCOM AI</div>
    <hr class="side-hr">
    <div class="side-title">About the Project</div>
    <div class="side-text">A machine learning application designed to predict customer churn
    and support customer retention decisions.</div>
    <hr class="side-hr">
    <div class="side-title">Model Information</div>
    <div class="side-label">Algorithm</div><div class="side-value">{algo_name}</div>
    <div class="side-label">Task</div><div class="side-value">Binary Classification</div>
    <div class="side-label">Target Variable</div><div class="side-value">Customer Churn</div>
    <hr class="side-hr">
    <div class="side-foot">Telcom Customer Churn Classification</div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>Customer Churn Prediction</h1>
  <p>Analyze customer information and predict the likelihood of customer churn using a
  trained Machine Learning model. Enter the customer details below to generate a prediction.</p>
</div>
<div class="page-h2">Customer Information</div>
""", unsafe_allow_html=True)

yn = ["Yes", "No"]
addon = ["Yes", "No", "No internet service"]

st.markdown('<div class="section">Personal Details</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2, gap="large")
with c1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", yn)
with c2:
    dependents = st.selectbox("Dependents", yn)
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12, step=1)


st.markdown('<div class="section">Services Information</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2, gap="large")
with c1:
    phone = st.selectbox("Phone Service", yn)
    multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", addon)
    tv = st.selectbox("Streaming TV", addon)
with c2:
    backup = st.selectbox("Online Backup", addon)
    protection = st.selectbox("Device Protection", addon)
    support = st.selectbox("Tech Support", addon)
    st.markdown("<div style='height:5.6rem'></div>", unsafe_allow_html=True)  
    movies = st.selectbox("Streaming Movies", addon)


st.markdown('<div class="section">Contract and Billing</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2, gap="large")
with c1:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", yn)
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
with c2:
    monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=1.0, format="%.2f")
    total = st.number_input("Total Charges", min_value=0.0, value=840.0, step=1.0, format="%.2f")

predict = st.button("Predict Customer Churn")


if predict:
    raw = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": multiple,
        "InternetService": internet, "OnlineSecurity": security,
        "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])
    processed_input = preprocess(raw)
    scaled_input = scaler.transform(processed_input)
    churn_p = float(model.predict_proba(scaled_input)[0, 1])
    stay_p = 1 - churn_p
    likely = churn_p >= THRESHOLD

    st.markdown('<div class="page-h2" style="margin-top:3rem">Prediction Result</div>',
                unsafe_allow_html=True)
    if likely:
        st.markdown("""<div class="result-card"><h3>Customer Likely to Churn</h3>
        <p>The model predicts that this customer is likely to leave the company.</p></div>""",
                    unsafe_allow_html=True)
    else:
        st.markdown("""<div class="result-card stay"><h3>Customer Likely to Stay</h3>
        <p>The model predicts that this customer is likely to remain with the company.</p></div>""",
                    unsafe_allow_html=True)

    st.markdown('<div class="page-h2" style="font-size:1.5rem;margin:2rem 0 1rem 0">Prediction Probability</div>',
                unsafe_allow_html=True)
    m1, m2 = st.columns(2, gap="medium")
    m1.markdown(f'<div class="metric-card"><div class="lbl">Churn Probability</div>'
                f'<div class="val">{churn_p:.2%}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><div class="lbl">Stay Probability</div>'
                f'<div class="val">{stay_p:.2%}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bar-label">Churn Probability</div>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{churn_p*100:.1f}%"></div></div>',
                unsafe_allow_html=True)

st.markdown('<div class="footer">Telcom Customer Churn Classification Project</div>',
            unsafe_allow_html=True)
