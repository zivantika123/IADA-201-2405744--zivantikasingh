import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from pathlib import Path

# --- Page Config ---
st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
)

# --- Custom CSS for modern green glassmorphism look with black table and white sidebar/drag box text ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&display=swap');
        html, body, .stApp {
            background: #111 !important;
            font-family: 'Montserrat', sans-serif !important;
            color: #fff !important;
        }
        .stFileUploader, .stFileUploader label, .stFileUploader span, .stFileUploader div {
            color: #fff !important;
        }
        .stButton, .stTextInput, .stSelectbox, .stSlider, .stProgress {
            background: rgba(255,255,255,0.0) !important;
            box-shadow: none !important;
            color: #fff !important;
        }
        .result-card {
            background: rgba(30,30,30,0.85);
            padding: 36px 28px 28px 28px;
            border-radius: 24px;
            box-shadow: 0 8px 32px 0 rgba(0,0,0,0.25);
            text-align: center;
            margin-top: 32px;
            color: #fff !important;
            backdrop-filter: blur(8px);
            border: 1.5px solid #333;
        }
        .bin-box {
            font-size: 34px;
            font-weight: bold;
            padding: 18px;
            border-radius: 18px;
            margin: 24px auto 18px auto;
            color: #fff;
            width: 280px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.10);
            letter-spacing: 1px;
        }
        .confidence, .waste-type, .footer, h1, h2, h3, h4, h5, h6, label, p, span, div, ul, li {
            color: #fff !important;
        }
        /* Table styles */
        table {
            margin: 0 auto 12px auto;
            border-collapse: separate;
            border-spacing: 0 2px;
            background: #222 !important;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.10);
        }
        th, td {
            font-size: 18px;
            padding: 10px 24px 10px 0;
            color: #fff !important;
            background: #222 !important;
        }
        th {
            font-weight: 700;
        }
        /* Sidebar history styles */
        .sidebar-history, .sidebar-history * {
            background: transparent !important;
            color: #fff !important;
        }
        .uploaded-img {
            border-radius: 20px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.10);
            margin-bottom: 20px;
        }
        .st-emotion-cache-1v0mbdj {
            padding-top: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load model ---
MODEL_PATH = "waste_mobilenetv2.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.error("Please ensure model file exists and is compatible")
    st.stop()

# --- Load class names ---
CLASS_NAMES_FILE = "class_names.json"
if Path(CLASS_NAMES_FILE).exists():
    class_names = json.loads(Path(CLASS_NAMES_FILE).read_text())
else:
    class_names = ['biodegradable', 'hazardous', 'recyclable']
    st.warning(f"{CLASS_NAMES_FILE} not found — using fallback {class_names}")

# --- Bin mapping (add all your classes here) ---
bin_map = {
    'biological': {
        'category': 'Biodegradable',
        'bin_name': 'Green Bin',
        'emoji': '🟢',
        'hex': '#28a745',
        'desc': 'Organic waste like food scraps, leaves, etc.'
    },
    'cardboard': {
        'category': 'Recyclable',
        'bin_name': 'Blue Bin',
        'emoji': '🔵',
        'hex': '#007bff',
        'desc': 'Recyclable: cardboard, paperboard, etc.'
    },
    'glass': {
        'category': 'Recyclable',
        'bin_name': 'Blue Bin',
        'emoji': '🔵',
        'hex': '#007bff',
        'desc': 'Recyclable: glass bottles, jars, etc.'
    },
    'metal': {
        'category': 'Recyclable',
        'bin_name': 'Blue Bin',
        'emoji': '🔵',
        'hex': '#007bff',
        'desc': 'Recyclable: cans, tins, metal items.'
    },
    'paper': {
        'category': 'Recyclable',
        'bin_name': 'Blue Bin',
        'emoji': '🔵',
        'hex': '#007bff',
        'desc': 'Recyclable: paper, magazines, books.'
    },
    'plastic': {
        'category': 'Recyclable',
        'bin_name': 'Blue Bin',
        'emoji': '🔵',
        'hex': '#007bff',
        'desc': 'Recyclable: plastic bottles, containers, etc.'
    },
    'trash': {
        'category': 'Non-Recyclable',
        'bin_name': 'Gray Bin',
        'emoji': '⚫',
        'hex': '#757575',
        'desc': 'General waste: not recyclable or hazardous.'
    },
    'clothes': {
        'category': 'Recyclable',
        'bin_name': 'Blue Bin',
        'emoji': '🔵',
        'hex': '#007bff',
        'desc': 'Recyclable: old clothes, textiles.'
    },
    'shoes': {
        'category': 'Recyclable',
        'bin_name': 'Blue Bin',
        'emoji': '🔵',
        'hex': '#007bff',
        'desc': 'Recyclable: shoes, footwear.'
    },
    'battery': {
        'category': 'Hazardous',
        'bin_name': 'Red Bin',
        'emoji': '🔴',
        'hex': '#dc3545',
        'desc': 'Hazardous: batteries, e-waste, chemicals.'
    }
}

# --- Preprocessing ---
def preprocess_image(img: Image.Image):
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# --- Header ---
st.markdown("<h1 style='text-align:center; font-size:2.7rem; letter-spacing:2px; color:#20703a;'>♻️ Smart Waste Classification System</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; font-size:1.4rem; color:#20703a;'>Detect waste type and get the right bin recommendation instantly</h3>", unsafe_allow_html=True)

# --- Sidebar: History ---
if "history" not in st.session_state:
    st.session_state["history"] = []

with st.sidebar:
    st.markdown("<div class='sidebar-history'>", unsafe_allow_html=True)
    st.header("🕓 Prediction History")
    if st.session_state["history"]:
        for i, entry in enumerate(reversed(st.session_state["history"]), 1):
            st.markdown(f"**{i}. {entry['waste_type']}** ({entry['category']})<br>"
                        f"<span style='font-size:13px;'>Bin: {entry['bin_name']} | Confidence: {entry['confidence']:.2f}%</span>",
                        unsafe_allow_html=True)
    else:
        st.write("No predictions yet.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Main Upload and Prediction ---
st.markdown("<br>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload an image of waste", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True, output_format="PNG", channels="RGB")

    # Predict
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    probs = predictions[0]
    top_idx = np.argmax(probs)
    top_label = class_names[top_idx]
    top_conf = float(probs[top_idx] * 100)

    # Bin info
    bin_info = bin_map.get(top_label.lower())
    if bin_info:
        bin_name = bin_info['bin_name']
        bin_emoji = bin_info['emoji']
        bin_color = bin_info['hex']
        category = bin_info['category']
        desc = bin_info['desc']
    else:
        bin_name = "Unknown"
        bin_emoji = "❓"
        bin_color = "#bbb"
        category = "Unknown"
        desc = "No bin mapping found for this category."

    # Save to history
    st.session_state["history"].append({
        "waste_type": top_label.capitalize(),
        "category": category,
        "bin_name": bin_name,
        "confidence": top_conf
    })

    # --- Result Card ---
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='bin-box' style='background:{bin_color}'>"
        f"{bin_emoji} {bin_name}</div>",
        unsafe_allow_html=True
    )

    # --- Table for details ---
    st.markdown(
        f"""
        <table>
            <tr><th style='text-align:left;'>Waste Type</th><td>{top_label.capitalize()}</td></tr>
            <tr><th style='text-align:left;'>Category</th><td>{category}</td></tr>
            <tr><th style='text-align:left;'>Bin Recommendation</th><td>{bin_emoji} {bin_name}</td></tr>
            <tr><th style='text-align:left;'>Description</th><td>{desc}</td></tr>
            <tr><th style='text-align:left;'>Confidence</th><td>{top_conf:.2f}%</td></tr>
        </table>
        """,
        unsafe_allow_html=True
    )

    # --- Top-3 Confidence Scores in a black table ---
    top3_indices = np.argsort(probs)[::-1][:3]
    st.markdown("""
    <table>
        <tr>
            <th style='text-align:left;'>Class</th>
            <th style='text-align:left;'>Confidence</th>
        </tr>
    """ + "".join(
        f"<tr><td>{class_names[i].capitalize()}</td><td>{probs[i]*100:.2f}%</td></tr>"
        for i in top3_indices
    ) + "</table>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # --- Clear history button ---
    if st.button("Clear History"):
        st.session_state["history"] = []

st.markdown("""
    <div class='footer'>
        <p class='footer-text'>Made by Pravar Golecha</p>
    </div>
""", unsafe_allow_html=True)
