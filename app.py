import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "Toxic_LSTM_Model.keras"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
HISTORY_PATH = "history.pkl"
label_cols = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

# ===============================
# LOAD MODEL & VECTORIZER
# ===============================
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        return None
@st.cache_resource
def load_vectorizer():
    if os.path.exists(VECTORIZER_PATH):
        with open(VECTORIZER_PATH, "rb") as f:
            return pickle.load(f)
    else:
        st.error(f"❌ Vectorizer file not found: {VECTORIZER_PATH}")
        return None
model = load_model()
vectorizer = load_vectorizer()

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="🌈 Toxic Comment Detector", page_icon="🤖", layout="wide")
st.markdown(
    '''
    <style>
    .main { background-color: #fffdf7; }
    .stTextInput>div>div>input { background-color: #f0f8ff; color: #000; font-weight: bold; }
    .stButton>button { background-color: #ff69b4; color: white; font-size: 16px; border-radius: 8px; }
    .stDataFrame { background-color: #f8fbff; }
    </style>
    ''',
    unsafe_allow_html=True
)
st.title("🌈 Toxic Comment Classification App")
st.subheader("🔥 Detect toxic comments in real-time and analyze model performance!")

# ===============================
# SIDEBAR MENU
# ===============================
st.sidebar.header("⚙️ Options")
option = st.sidebar.radio("Choose Action:", ["Real-time Prediction", "Bulk Prediction", "Insights & Metrics"])

# ===============================
# REAL-TIME PREDICTION
# ===============================
if option == "Real-time Prediction":
    st.markdown("### ✍️ Enter a comment below and check its toxicity:")
    user_input = st.text_area("Enter Comment:", "")
    if st.button("Predict 🚀") and model and vectorizer:
        if user_input.strip() != "":
            X_input = vectorizer.transform([user_input])
            pred = model.predict(X_input)[0]
            results = dict(zip(label_cols, pred))
            st.success("✅ Prediction Complete!")
            df_res = pd.DataFrame(results.items(), columns=["Category", "Probability"])
            df_res["Probability"] = df_res["Probability"].round(3)
            st.dataframe(df_res)
            # Visualization
            fig, ax = plt.subplots(figsize=(8,4))
            sns.barplot(x="Category", y="Probability", data=df_res, palette="coolwarm", ax=ax)
            ax.set_ylim(0,1)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("⚠️ Please enter a comment!")

# ===============================
# BULK PREDICTION
# ===============================
elif option == "Bulk Prediction":
    st.markdown("### 📂 Upload a CSV file")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None and model and vectorizer:
        df = pd.read_csv(uploaded_file)
        if "comment_text" in df.columns:
            # 🔍 Data Insights
            st.markdown("#### 📊 Data Insights")
            st.write(df.describe(include="all"))
            st.write("Dataset Shape:", df.shape)
            st.write("First few comments:")
            st.write(df["comment_text"].head())
            # Transform with vectorizer
            X_input = vectorizer.transform(df["comment_text"].astype(str).tolist())
            preds = model.predict(X_input)
            preds_df = pd.DataFrame(preds, columns=label_cols)
            result = pd.concat([df, preds_df], axis=1)
            st.success("✅ Bulk Predictions Done!")
            st.dataframe(result.head(10))
            # Download Option
            csv = result.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results CSV", csv, "predictions.csv", "text/csv")
        else:
            st.error("❌ CSV must have a 'comment_text' column!")

# ===============================
# INSIGHTS & METRICS
# ===============================
elif option == "Insights & Metrics":
    st.markdown("### 📊 Model Insights & Performance Metrics")
    # Load training history if available
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "rb") as f:
            history = pickle.load(f)
        st.write("#### 📈 Training History")
        fig, ax = plt.subplots(1,2, figsize=(12,4))
        ax[0].plot(history.get("accuracy", []), label="Train Acc")
        ax[0].plot(history.get("val_accuracy", []), label="Val Acc")
        ax[0].set_title("Accuracy")
        ax[0].legend()
        ax[1].plot(history.get("loss", []), label="Train Loss")
        ax[1].plot(history.get("val_loss", []), label="Val Loss")
        ax[1].set_title("Loss")
        ax[1].legend()
        st.pyplot(fig)
    else:
        st.info("ℹ️ Training history not found. Save 'history.pkl' for detailed metrics.")
    # Sample test cases
    if model and vectorizer:
        st.markdown("#### 🔍 Sample Test Cases")
        sample_texts = [
            "I love this community ❤️",
            "You are an idiot 🤬",
            "Let's meet for coffee tomorrow ☕",
            "This is the worst thing ever 😡"
        ]
        X_input = vectorizer.transform(sample_texts)
        preds = model.predict(X_input)
        sample_df = pd.DataFrame(preds, columns=label_cols)
        sample_df.insert(0, "Comment", sample_texts)
        st.dataframe(sample_df.round(3))
