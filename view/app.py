# app.py
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import io
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set page configuration
st.set_page_config(
    page_title="BrainScan - Détection Cancer Cérébral",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto"
)

# ---------- Helpers ----------
@st.cache_resource(show_spinner=False)
def load_cnn_model(path="./models/best_cnn_model.h5"):
    return load_model(path)

def preprocess_image(pil_img, target_size=(224, 224)):
    img = ImageOps.fit(
        pil_img.convert("RGB"),
        target_size,
        method=Image.Resampling.LANCZOS
    )
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def predict_image(model, pil_img):
    x = preprocess_image(pil_img)
    preds = model.predict(x)
    probs = preds.flatten()
    top_idx = int(np.argmax(probs))
    return top_idx, probs

def plot_probs(probs, labels=None):
    if labels is None:
        labels = ["Gliome", "Méningiome", "Normal", "Pituitaire"]
    df = pd.DataFrame({"label": labels, "prob": probs})
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(df["label"], df["prob"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilité")
    ax.set_title("Probabilités par classe")
    plt.xticks(rotation=25)
    plt.tight_layout()
    return fig, df

# ---------- Configuration ----------
st.set_page_config(
    page_title="BrainScan - Détection Cancer Cérébral",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto",
)

# Initialize session state
if "model" not in st.session_state:
    st.session_state["model"] = None

# Constants
LABELS = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_PATH = "./models/best_cnn_model.h5"

# Load model at startup
if st.session_state["model"] is None:
    try:
        st.session_state["model"] = load_cnn_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Erreur de chargement du modèle : {e}")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("ℹ️ À propos")
    st.write("BrainScan — interface de démo pour la détection du cancer cérébral à partir d'IRM.")
    st.markdown("---")

    st.header("📖 Documentation du Projet")
    st.write("""
    BrainScan est un projet de détection du cancer cérébral utilisant un CNN (Convolutional Neural Network) 
    pour analyser les images IRM. Le modèle a été entraîné sur quatre classes différentes :
    - Gliome
    - Méningiome
    - Absence de tumeur
    - Tumeur pituitaire
    """)
    
    st.markdown("---")
    st.subheader("🎯 Objectif")
    st.write("""
    Faciliter et accélérer le diagnostic préliminaire des tumeurs cérébrales 
    en utilisant l'intelligence artificielle comme outil d'aide à la décision.
    """)
    
    st.markdown("---")
    st.subheader("🔬 Caractéristiques techniques")
    st.write("""
    - Architecture : CNN personnalisé
    - Taille d'entrée : 224×224 pixels
    - Normalisation : [0,1]
    - Sortie : Probabilités pour 4 classes
    """)

    st.markdown("---")
    st.write("Version: 1.0")
    st.write("&copy; 2025 Team BrainScan")

# ---------- Main content ----------
# Custom CSS for professional design
st.markdown("""
<style>
.main-title {
    color: #2C3E50;
    font-size: 2.8rem !important;
    font-weight: 700;
    margin-bottom: 1.5rem !important;
    text-align: center;
    letter-spacing: -0.5px;
}
.main-subtitle {
    color: #34495E;
    font-size: 1.2rem !important;
    font-weight: 400;
    margin-bottom: 2rem !important;
    text-align: center;
}
.upload-container {
    background: linear-gradient(145deg, #ffffff 0%, #f8faff 100%);
    border-radius: 24px;
    padding: 3rem 2rem;
    text-align: center;
    margin: 2rem 0;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    border: 1px solid rgba(0, 0, 0, 0.05);
}
.results-card {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
    border: 1px solid rgba(0, 0, 0, 0.05);
}
.prediction-label {
    font-size: 1.8rem;
    color: #2C3E50;
    margin: 1.5rem 0;
    font-weight: 600;
    text-align: center;
}
.prediction-type {
    font-size: 2.2rem;
    color: #3498DB;
    margin: 1rem 0;
    font-weight: 700;
    text-align: center;
}
.details-section {
    background: #f8faff;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.stButton button {
    background: linear-gradient(135deg, #3498DB 0%, #2980B9 100%) !important;
    color: white !important;
    border-radius: 50px !important;
    padding: 0.8rem 2.5rem !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3) !important;
    transition: all 0.3s ease !important;
}
.stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4) !important;
}
div.stDataFrame {
    padding: 1.5rem;
    border-radius: 12px;
    background: #ffffff;
    border: 1px solid rgba(0, 0, 0, 0.05);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
}
.footer {
    background: linear-gradient(to right, #2C3E50, #3498DB);
    padding: 1rem;
    text-align: center;
    position: fixed;
    bottom: 0;
    width: 100%;
    box-shadow: 0 -4px 20px rgba(0,0,0,0.1);
    color: white !important;
}
.footer p {
    color: white !important;
    opacity: 0.9;
}
.upload-prompt {
    font-size: 1.4rem;
    color: #34495E;
    margin: 1rem 0;
    font-weight: 500;
}
.upload-subtitle {
    color: #7F8C8D;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# Title with modern styling
st.markdown('<h1 class="main-title">🧠 BrainScan</h1>', unsafe_allow_html=True)

# Main container
main_container = st.container()

with main_container:
    # Upload section with gradient background
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if not uploaded_file:
        st.markdown("### Déposez votre image IRM ici")
        st.markdown("*Format accepté : JPG, PNG*")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        # Create two columns for results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="results-card">', unsafe_allow_html=True)
            img = Image.open(uploaded_file)
            st.image(img, caption="Image IRM analysée", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="results-card">', unsafe_allow_html=True)
            model = st.session_state.get("model")
            
            if model is None:
                try:
                    model = load_cnn_model(MODEL_PATH)
                    st.session_state["model"] = model
                except Exception as e:
                    st.error(f"Impossible de charger le modèle : {e}")
            
            if model is not None:
                with st.spinner("Analyse en cours..."):
                    class_idx, probs = predict_image(model, img)
                
                predicted_label = LABELS[class_idx]
                st.markdown(f'<p class="prediction-label">Diagnostic : {predicted_label}</p>', unsafe_allow_html=True)
                
                # Probability chart
                fig, df_probs = plot_probs(probs)
                st.pyplot(fig)
                
                # Probabilities table
                st.markdown("### Détails des probabilités")
                st.dataframe(
                    df_probs.sort_values("prob", ascending=False)
                    .reset_index(drop=True)
                    .style.format({"prob": "{:.2%}"})
                )
                
                # Export button
                result = {"predicted_label": predicted_label}
                result.update({lbl: float(p) for lbl, p in zip(LABELS, probs)})
                buf = io.BytesIO()
                pd.DataFrame([result]).to_csv(buf, index=False)
                buf.seek(0)
                
                st.download_button(
                    "⬇️ Télécharger le rapport",
                    data=buf,
                    file_name="rapport_analyse.csv",
                    mime="text/csv"
                )
            st.markdown("</div>", unsafe_allow_html=True)

# Footer with subtle shadow
st.markdown("""
<div style='
    position: fixed;
    bottom: 0;
    width: 100%;
    background-color: rgba(248, 249, 250, 0.95);
    padding: 0.5rem;
    text-align: center;
    box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
    backdrop-filter: blur(5px);
'>
    <p style='margin: 0;'>© 2025 Team BrainScan - Tous droits réservés</p>
</div>
""", unsafe_allow_html=True)
