import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# HEADER
# ==========================================
st.markdown("""
<h1 style='text-align:center;color:#4CAF50;'>
🎓 Student Performance Prediction System
</h1>
<h4 style='text-align:center;color:gray;'>
Machine Learning Based Student Pass/Fail Predictor
</h4>
<hr>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():

    df = pd.read_csv("student-mat 1.csv")

    # Create Target
    df["Pass"] = (df["G3"] >= 10).astype(int)

    # Keep only required columns
    df = df[[
        "age",
        "studytime",
        "failures",
        "absences",
        "internet",
        "Pass"
    ]]

    # Encode internet column
    le = LabelEncoder()
    df["internet"] = le.fit_transform(df["internet"])

    return df

df = load_data()

# ==========================================
# TRAIN MODEL
# ==========================================
X = df.drop("Pass", axis=1)
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# ==========================================
# SIDEBAR INPUTS
# ==========================================
st.sidebar.header("📋 Student Details")

age = st.sidebar.slider(
    "Age",
    15,
    22,
    17
)

studytime = st.sidebar.selectbox(
    "Study Time",
    [1, 2, 3, 4]
)

failures = st.sidebar.selectbox(
    "Previous Failures",
    [0, 1, 2, 3]
)

absences = st.sidebar.slider(
    "Absences",
    0,
    50,
    5
)

internet = st.sidebar.selectbox(
    "Internet Access",
    ["Yes", "No"]
)

# ==========================================
# DASHBOARD METRICS
# ==========================================
col1, col2, col3 = st.columns(3)

col1.metric("📊 Total Records", len(df))
col2.metric("🎯 Features", X.shape[1])
col3.metric("✅ Accuracy", f"{accuracy*100:.2f}%")

st.divider()

# ==========================================
# PREDICTION
# ==========================================
if st.button("🚀 Predict Result", use_container_width=True):

    input_df = pd.DataFrame({
        "age": [age],
        "studytime": [studytime],
        "failures": [failures],
        "absences": [absences],
        "internet": [1 if internet == "Yes" else 0]
    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    left, right = st.columns([2, 1])

    with left:

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("✅ Student is likely to PASS")
        else:
            st.error("❌ Student is likely to FAIL")

        st.write(
            f"Pass Probability: **{probability[1]*100:.2f}%**"
        )

    with right:

        fig, ax = plt.subplots()

        labels = ["Fail", "Pass"]
        values = [probability[0], probability[1]]

        ax.bar(labels, values)

        ax.set_ylabel("Probability")

        st.pyplot(fig)

# # ==========================================
# # DATASET PREVIEW
# # ==========================================
# st.subheader("📄 Dataset Preview")

# st.dataframe(df.head())

# # ==========================================
# # STATISTICS
# # ==========================================
# st.subheader("📈 Dataset Statistics")

# st.dataframe(df.describe())

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<hr>
<center>
🎓 Developed using Streamlit + Logistic Regression
</center>
""", unsafe_allow_html=True)


st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1518770660439-4636190af475");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
</style>
""", unsafe_allow_html=True)
