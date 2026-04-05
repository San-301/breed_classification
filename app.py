import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import time
from PIL import Image

# ==========================================
# 1. SYSTEM CONFIGURATION & DATA
# ==========================================
MODEL_PATH = "breed_classifier_mobilenet(2).h5"
CONFIDENCE_THRESHOLD = 0.58  # Optimized rejection threshold

# Ensure retraining directory exists
if not os.path.exists("flagged_for_learning"):
    os.makedirs("flagged_for_learning")

BREED_DATA = {
    "Bhadawari": {"Type": "Buffalo", "Origin": "UP & MP", "Description": "High-fat milk breed, heat-tolerant."},
    "Gir": {"Type": "Cattle", "Origin": "Gujarat", "Description": "High milk-yielding indigenous breed."},
    "Jaffarabadi": {"Type": "Buffalo", "Origin": "Gujarat", "Description": "Large-sized dairy buffalo."},
    "Kankrej": {"Type": "Cattle", "Origin": "Gujarat & Rajasthan", "Description": "Dual-purpose breed for milk and draught."},
    "Murrah": {"Type": "Buffalo", "Origin": "Haryana & Punjab", "Description": "High-fat milk producing buffalo."},
    "Nagpuri": {"Type": "Buffalo", "Origin": "Maharashtra", "Description": "Drought-resistant dairy buffalo."},
    "Ongole": {"Type": "Cattle", "Origin": "Andhra Pradesh", "Description": "Large breed used for draught and milk."},
    "Red_Sindhi": {"Type": "Cattle", "Origin": "Sindh region", "Description": "Adapted for tropical climates."},
    "Sahiwal": {"Type": "Cattle", "Origin": "Punjab", "Description": "Heat-tolerant dairy breed."},
    "Toda": {"Type": "Buffalo", "Origin": "Nilgiri Hills", "Description": "Small-sized buffalo, adapted to hilly terrain."}
}
CLASS_NAMES = sorted(BREED_DATA.keys())

# ==========================================
# 2. UI STYLING
# ==========================================
st.set_page_config(page_title="Bovine Intel Pro", layout="wide", page_icon="🐂")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stSidebar { background-color: #111; color: white; }
    .stButton>button { background-color: #1e40af; color: white; border-radius: 10px; height: 3.5em; font-weight: bold; }
    .result-card { 
        background: white; 
        padding: 25px; 
        border-radius: 15px; 
        border-left: 5px solid #1e40af; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

def predict_breed(img, model):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    predictions = model.predict(img_array)
    score = np.max(predictions)
    class_idx = np.argmax(predictions)
    
    return CLASS_NAMES[class_idx], score

# ==========================================
# 4. MAIN APP LOGIC
# ==========================================
st.title("🐂 Bovine Intel Pro")
st.subheader("Advanced Cattle & Buffalo Breed Classification")

model = load_model()

if model is None:
    st.error(f"Model file '{MODEL_PATH}' not found. Please upload it to the repository.")
else:
    # --- FIXED SECTION ---
    input_method = st.radio(
        "Select Input Method:", 
        options=["Upload Image", "Use Camera"], 
        horizontal=True
    )
    # ---------------------

    uploaded_file = None
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Take a photo of the animal")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Target Bovine", width=400)
        
        if st.button("🚀 Analyze Breed"):
            with st.spinner("Running deep analysis..."):
                breed, confidence = predict_breed(img, model)
                
                if confidence < CONFIDENCE_THRESHOLD:
                    st.warning("⚠️ Low Confidence. This may not be a recognized breed.")
                    # Flag for learning
                    img.save(f"flagged_for_learning/unknown_{int(time.time())}.jpg")
                else:
                    details = BREED_DATA[breed]
                    st.markdown(f"""
                        <div class="result-card">
                            <h2 style="color:#1e40af;">Result: {breed}</h2>
                            <p><b>Confidence Score:</b> {confidence:.2%}</p>
                            <hr>
                            <p><b>Type:</b> {details['Type']}</p>
                            <p><b>Origin:</b> {details['Origin']}</p>
                            <p><b>Description:</b> {details['Description']}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
