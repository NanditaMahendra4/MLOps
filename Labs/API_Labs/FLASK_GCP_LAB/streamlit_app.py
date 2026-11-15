import streamlit as st
import requests

API_URL = "https://iris-app-125862335227.us-east1.run.app/predict"

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Wine Classifier",
    page_icon="🍷",
    layout="centered",
    initial_sidebar_state="expanded",
)

# 2. CUSTOM CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #722F37;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #5A252C;
        border-color: #5A252C;
        color: white;
    }
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# 3. FEATURE DEFINITIONS: (api_key, label, min, max, default, step)
# Defaults are a real class_0 sample from the dataset.
FEATURES = [
    ("alcohol",              "Alcohol (%)",             10.0,  16.0,   14.23, 0.01),
    ("malic_acid",           "Malic Acid",               0.0,   7.0,    1.71, 0.01),
    ("ash",                  "Ash",                      1.0,   4.0,    2.43, 0.01),
    ("alcalinity_of_ash",    "Alcalinity of Ash",       10.0,  32.0,   15.6,  0.1),
    ("magnesium",            "Magnesium",               60.0, 170.0,  127.0,  1.0),
    ("total_phenols",        "Total Phenols",            0.5,   4.0,    2.8,   0.01),
    ("flavanoids",           "Flavanoids",               0.0,   5.5,    3.06,  0.01),
    ("nonflavanoid_phenols", "Nonflavanoid Phenols",     0.0,   1.0,    0.28,  0.01),
    ("proanthocyanins",      "Proanthocyanins",          0.0,   4.0,    2.29,  0.01),
    ("color_intensity",      "Color Intensity",          1.0,  14.0,    5.64,  0.01),
    ("hue",                  "Hue",                      0.4,   2.0,    1.04,  0.01),
    ("od280_od315",          "OD280/OD315 of Diluted Wines", 1.0, 4.5,  3.92,  0.01),
    ("proline",              "Proline",                200.0, 1800.0, 1065.0,  1.0),
]

# Model output -> display name and emoji
CLASS_INFO = {
    "class_0": ("Cultivar 1", "🍇"),
    "class_1": ("Cultivar 2", "🍷"),
    "class_2": ("Cultivar 3", "🥂"),
}

# 4. SIDEBAR
with st.sidebar:
    st.title("🍷 About the App")
    st.info(
        """
        This machine learning app predicts the **cultivar** (grape variety) of a wine
        from 13 chemical measurements.

        **Dataset:** scikit-learn Wine dataset (178 samples, 3 classes)

        **Model:** Support Vector Machine (RBF kernel) with feature scaling
        """
    )
    st.write("---")
    st.caption("Built with Streamlit & Cloud Run")

# 5. MAIN INTERFACE
st.title("🍷 Wine Cultivar Predictor")
st.markdown("Adjust the sliders below to enter the wine's chemical analysis.")

values = {}
cols = st.columns(3)
for i, (key, label, lo, hi, default, step) in enumerate(FEATURES):
    with cols[i % 3]:
        values[key] = st.slider(label, lo, hi, default, step=step)

st.write("---")

# 6. PREDICTION LOGIC
if st.button("🔍 Predict Cultivar"):
    with st.spinner("Analyzing wine data..."):
        try:
            response = requests.post(API_URL, json=values, timeout=30)

            if response.status_code == 200:
                result = response.json()
                raw_label = result["prediction"]
                confidence = result.get("confidence")
                name, emoji = CLASS_INFO.get(raw_label, (raw_label, "🍷"))

                st.success("Prediction Complete!")

                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    st.markdown(f"<div style='font-size:90px;text-align:center'>{emoji}</div>",
                                unsafe_allow_html=True)
                with res_col2:
                    st.header(f"It's **{name}**!")
                    if confidence is not None:
                        st.metric("Model confidence", f"{confidence * 100:.1f}%")
                        st.progress(float(confidence))
                    st.caption(f"Raw model output: {raw_label}")
                st.balloons()

            elif response.status_code == 400:
                st.error(f"Bad request: {response.json().get('error', 'invalid input')}")
            else:
                st.error(f"Server Error: {response.status_code}")

        except requests.exceptions.RequestException:
            st.error("Connection Error: Could not reach the prediction service.")