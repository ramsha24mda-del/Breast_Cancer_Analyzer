import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, roc_curve, auc

# --- Page Config ---
st.set_page_config(
    page_title="Breast Cancer Analyzer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model ---
with open("model/cancer.pkl", "rb") as f:
    model = pickle.load(f)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
/* Main Title & Tagline */
.main-title { font-size:52px; color:#b30000; font-weight:800; text-align:center; margin-bottom:5px; }
.subtitle { font-size:20px; color:#333; text-align:center; margin-bottom:20px; }

/* Feature cards container */
.feature-container { display:flex; justify-content:center; flex-wrap:wrap; gap:25px; margin-top:30px; }

/* Individual feature card */
.feature-card {
    background: linear-gradient(135deg,#f0fff0,#e6ffe6);
    padding:25px;
    border-radius:20px;
    width:260px;
    box-shadow:0 6px 20px rgba(0,0,0,0.1);
    text-align:center;
    transition: transform 0.3s, box-shadow 0.3s;
}
.feature-card:hover { transform: translateY(-10px); box-shadow:0 12px 25px rgba(0,0,0,0.2); }

/* Icon wrapper */
.icon-wrapper {
    background:#e6ffe6;
    border-radius:50%;
    width:80px;
    height:80px;
    display:flex;
    align-items:center;
    justify-content:center;
    margin:auto;
    transition: transform 0.3s;
}
.feature-card:hover .icon-wrapper { transform: scale(1.2); }

/* Prediction Cards */
.prediction-card {
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}
.benign { background-color:#e6ffe6; color:#008000; }
.malignant { background-color:#ffe6e6; color:#b30000; }

</style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
page = st.sidebar.radio("📌 Navigate to:", ["🏠 Home", "🔮 Batch Prediction", "🧪 Single Prediction", "📊 Insights"])

# --- Home Page ---
# --- Home Page ---
if page == "🏠 Home":
    st.markdown("""
    <style>
    /* Container for all feature cards */
    .feature-container {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 25px;
        margin-top: 30px;
    }

    /* Individual feature cards */
    .feature-card {
        background: linear-gradient(135deg, #f0fff0, #e6ffe6);
        padding: 25px;
        border-radius: 20px;
        width: 260px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        transition: transform 0.3s, box-shadow 0.3s;
        text-align: center;
    }
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 12px 25px rgba(0,0,0,0.2);
    }

    /* Icon wrapper */
    .icon-wrapper {
        background: #ffffff;
        border-radius: 50%;
        width: 80px;
        height: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: auto;
    }

    /* Title of card */
    .feature-card h3 {
        margin-top: 15px;
        margin-bottom: 10px;
        font-size: 20px;
        font-weight: 700;
    }

    /* Description of card */
    .feature-card p {
        font-size: 15px;
        color: #555;
    }

    /* Main Header */
    .main-card {
        background: linear-gradient(135deg, #e6f0ff, #ffffff);
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    .main-title {
        font-size: 48px;
        color: #003366;
        font-weight: 800;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 20px;
        color: #444;
        margin-bottom: 20px;
    }
    </style>

    <div class="main-card">
        <h1 class="main-title">🩺 Breast Cancer Analyzer</h1>
        <p class="subtitle">AI-powered early detection & personalized risk analysis for better health</p>
        <div class="feature-container">
            <div class="feature-card">
                <div class="icon-wrapper">
                    <img src="https://img.icons8.com/color/96/000000/brain.png" width="50"/>
                </div>
                <h3>AI-Powered Model</h3>
                <p>Random Forest trained on Breast Cancer dataset for accurate predictions.</p>
            </div>
            <div class="feature-card">
                <div class="icon-wrapper">
                    <img src="https://img.icons8.com/color/96/000000/stethoscope.png" width="50"/>
                </div>
                <h3 style="color:#008000;">Instant Prediction</h3>
                <p>Detects tumor as Benign or Malignant in real-time using user inputs.</p>
            </div>
            <div class="feature-card">
                <div class="icon-wrapper">
                    <img src="https://img.icons8.com/color/96/000000/graph.png" width="50"/>
                </div>
                <h3 style="color:#b30000;">Visual Insights</h3>
                <p>Interactive charts for risk analysis, feature importance, and ROC curve.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# --- Batch Prediction ---
elif page == "🔮 Batch Prediction":
    st.header("🔮 Batch Prediction")
    st.markdown("Upload a CSV with same features as training data to predict Benign/Malignant tumors.")

    uploaded_file = st.file_uploader("📂 Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        X = df.copy()
        if "diagnosis" in X.columns:
            X = X.drop("diagnosis", axis=1)

        predictions = model.predict(X)
        pred_labels = ["Malignant" if p in [1,'M'] else "Benign" for p in predictions]
        df["Prediction"] = pred_labels

        # Risk summary
        malignant_count = pred_labels.count("Malignant")
        total_count = len(pred_labels)
        malignant_percent = round((malignant_count/total_count)*100,2)

        if malignant_percent == 0:
            risk_level = "Low Risk ✅"
        elif malignant_percent <= 30:
            risk_level = "Moderate Risk ⚠️"
        else:
            risk_level = "High Risk ❌"

        st.success(f"✅ Predictions Complete! Malignant Cases: {malignant_count}/{total_count} ({malignant_percent}%)")
        st.info(f"📊 Overall Risk Level: {risk_level}")
        st.dataframe(df)

        fig = px.pie(df, names="Prediction", title="Prediction Distribution",
                     color_discrete_sequence=px.colors.sequential.Reds)
        st.plotly_chart(fig)

# --- Single Prediction ---
elif page == "🧪 Single Prediction":
    st.header("🧪 Single Prediction Form (Top 5 Features)")
    st.markdown("Enter the top 5 important feature values:")

    col1, col2 = st.columns(2)
    with col1:
        mean_radius = st.number_input("Mean Radius (6-28)", min_value=0.0, max_value=40.0, value=14.0, step=0.1)
        mean_texture = st.number_input("Mean Texture (9-40)", min_value=0.0, max_value=50.0, value=19.0, step=0.1)
    with col2:
        mean_perimeter = st.number_input("Mean Perimeter (40-190)", min_value=0.0, max_value=250.0, value=90.0, step=0.5)
        mean_area = st.number_input("Mean Area (150-2500)", min_value=0.0, max_value=4000.0, value=600.0, step=10.0)
        mean_smoothness = st.number_input("Mean Smoothness (0.05-0.16)", min_value=0.0, max_value=1.0, value=0.1, format="%.5f")

    if st.button("🔍 Predict Tumor"):
        input_data = pd.DataFrame([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]],
                                  columns=["mean_radius","mean_texture","mean_perimeter","mean_area","mean_smoothness"])
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        result = "Malignant ❌" if prediction in [1,'M'] else "Benign ✅"
        if "Malignant" in result:
            st.markdown(f"<div class='prediction-card malignant'>🔴 Prediction: {result}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='prediction-card benign'>🟢 Prediction: {result}</div>", unsafe_allow_html=True)

        # Confidence bar
        proba_df = pd.DataFrame({"Class":["Benign","Malignant"], "Probability":proba})
        fig = px.bar(proba_df, x="Class", y="Probability", color="Class",
                     color_discrete_map={"Benign":"green","Malignant":"red"},
                     title="Prediction Confidence")
        fig.update_layout(yaxis=dict(range=[0,1], tickformat=".0%"))
        st.plotly_chart(fig)

# --- Insights Page ---
elif page == "📊 Insights":
    st.header("📊 Model Insights & Performance")

    df = pd.read_csv("Breast_cancer_data.csv")
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]
    y_pred = model.predict(X)

    # Accuracy
    accuracy = np.round((y_pred==y).mean()*100,2)
    st.subheader(f"✅ Model Accuracy: {accuracy}%")

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    fig_cm = ff.create_annotated_heatmap(cm, x=['Benign','Malignant'], y=['Benign','Malignant'],
                                        colorscale='Viridis', showscale=True)
    st.subheader("🔻 Confusion Matrix")
    st.plotly_chart(fig_cm)

    # Feature Importance
    importance = model.feature_importances_
    features = X.columns
    fi_df = pd.DataFrame({"Feature":features,"Importance":importance}).sort_values(by="Importance",ascending=False)
    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale="Viridis", title="🔥 Feature Importance")
    st.plotly_chart(fig_fi)

    # ROC Curve
    if hasattr(model,"predict_proba"):
        y_prob = model.predict_proba(X)[:,1]
        y_true = y.map({'B':0,'M':1}) if y.dtype=='O' else y
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr,tpr)

        fig_roc = px.area(x=fpr, y=tpr, title=f"📈 ROC Curve (AUC={roc_auc:.2f})",
                         labels=dict(x='False Positive Rate',y='True Positive Rate'))
        fig_roc.add_scatter(x=fpr, y=tpr, mode='lines', line=dict(color='red',width=3))
        fig_roc.add_shape(type='line', line=dict(dash='dash',color='black'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig_roc)

    # Correlation Heatmap
    corr = X.corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis', title="📊 Feature Correlation")
    st.plotly_chart(fig_corr)
